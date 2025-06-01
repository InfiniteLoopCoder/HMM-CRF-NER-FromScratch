import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import math
import wandb
from tqdm import tqdm
import functools

# Define special tokens and tags
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TAG = "<START>" # For CRF layer
STOP_TAG = "<STOP>"   # For CRF layer
PAD_TAG_ID = -1      # Special ID for padding in tags, ignored in loss calculation

# Default Hyperparameters (can be overridden by args or dynamically calculated)
DEFAULT_MAX_SEQ_LEN = 128
EMBEDDING_DIM = 256
HIDDEN_DIM = 512    # d_model for Transformer
N_HEADS = 8         # Number of attention heads
N_LAYERS = 3        # Number of Transformer encoder layers
DROPOUT = 0.1

def _get_next_power_of_2(n):
    """ Helper function to find the next power of 2, useful for sequence padding to optimize Transformer operations. """
    if n <= 0:
        return 1
    power = 1
    while power < n:
        power *= 2
    return power

class PositionalEncoding(nn.Module):
    """ Injects positional information into the input embeddings. """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe) # Not a model parameter

    def forward(self, x):
        """ x shape: [seq_len, batch_size, embedding_dim] """
        x = x + self.pe[:x.size(0), :] # Add positional encoding to input
        return self.dropout(x)

class TransformerCRF(nn.Module):
    """ Transformer Encoder followed by a CRF layer for sequence tagging. """
    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim, n_heads, n_layers, dropout, pad_idx, current_max_seq_len):
        super(TransformerCRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.num_tags = len(tag_to_idx)
        self.pad_idx = pad_idx # Index for PAD_TOKEN in vocabulary, used by embedding layer
        self.hidden_dim = hidden_dim # Corresponds to d_model in Transformer

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=current_max_seq_len + 5) # Add buffer to max_len

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Linear layer to map Transformer output to tag space (emissions)
        self.hidden2tag = nn.Linear(embedding_dim, self.num_tags)

        # CRF transition parameters. transitions[i, j] is the score of transitioning from tag_i to tag_j.
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        # IMPORTANT Exception:
        #  No transitions to START_TAG from any other tag.
        #  No transitions from STOP_TAG to any other tag.
        # These are handled by setting scores to a very small number.
        self.transitions.data[:, tag_to_idx[START_TAG]] = -10000.0
        self.transitions.data[tag_to_idx[STOP_TAG], :] = -10000.0


    def _get_transformer_emissions(self, sentences, attention_mask):
        """ Computes emission scores from the Transformer encoder. """
        # sentences shape: [seq_len, batch_size]
        # attention_mask shape: [batch_size, seq_len] (True for non-pad, False for pad)

        embedded = self.embedding(sentences) * math.sqrt(self.embedding.embedding_dim) # [seq_len, batch_size, emb_dim]
        pos_encoded = self.pos_encoder(embedded) # [seq_len, batch_size, emb_dim]

        # Transformer expects src_key_padding_mask: [batch_size, src_seq_len], True indicates padding.
        src_key_padding_mask = ~attention_mask # Invert our mask

        transformer_out = self.transformer_encoder(pos_encoded, src_key_padding_mask=src_key_padding_mask)
        # transformer_out shape: [seq_len, batch_size, emb_dim]

        emissions = self.hidden2tag(transformer_out) # [seq_len, batch_size, num_tags]
        return emissions

    def _calculate_true_path_score(self, emissions, tags, attention_mask_seq_first):
        """ Calculates the score of the true tag sequence. """
        # emissions: [seq_len, batch_size, num_tags]
        # tags: [seq_len, batch_size] (true tag indices)
        # attention_mask_seq_first: [seq_len, batch_size] (True for non-pad)

        batch_size = emissions.size(1)
        seq_len = emissions.size(0)
        score = torch.zeros(batch_size, device=emissions.device)

        # Add emission scores from Transformer output
        for i in range(batch_size):
            true_len = attention_mask_seq_first[:, i].sum().int()
            # Sum scores for non-padded tokens based on their true tags.
            score[i] += emissions[torch.arange(true_len), i, tags[torch.arange(true_len), i]].sum()

        # Add transition scores from CRF layer
        start_tag_tensor = torch.full((1, batch_size), self.tag_to_idx[START_TAG], dtype=torch.long, device=tags.device)
        tags_with_start = torch.cat([start_tag_tensor, tags], dim=0)

        for i in range(batch_size):
            true_len = attention_mask_seq_first[:, i].sum().int()
            # Iterate through transitions for the true sequence length
            for t in range(true_len): # transition from tags_with_start[t] to tags_with_start[t+1]
                score[i] += self.transitions[tags_with_start[t, i], tags_with_start[t+1, i]]
            # Add transition from last true tag to STOP_TAG
            score[i] += self.transitions[tags_with_start[true_len, i], self.tag_to_idx[STOP_TAG]]

        return score


    def _calculate_log_partition_function(self, emissions, attention_mask_seq_first):
        """ Calculates the log partition function (log Z(x)) using the forward algorithm. """
        # emissions: [seq_len, batch_size, num_tags]
        # attention_mask_seq_first: [seq_len, batch_size] (True for non-pad)

        seq_len = emissions.size(0)
        batch_size = emissions.size(1)

        # log_alpha[b, j] = log probability of paths ending at current word, batch b, with tag j
        log_alpha = torch.full((batch_size, self.num_tags), -10000.0, device=emissions.device)
        log_alpha[:, self.tag_to_idx[START_TAG]] = 0.0 # Initialize: score of being in START_TAG is log(1)=0.

        for t in range(seq_len):
            current_emissions = emissions[t] # [batch_size, num_tags]
            log_alpha_t = torch.full((batch_size, self.num_tags), -10000.0, device=emissions.device)

            for next_tag_idx in range(self.num_tags):
                # For paths ending at word t with tag `next_tag_idx`:
                # Sum over all `prev_tag_idx`:
                #   `log_alpha[:, prev_tag_idx]` (score of path ending at t-1 with `prev_tag_idx`)
                # + `self.transitions[prev_tag_idx, next_tag_idx]` (score of transition)
                # Then logsumexp over `prev_tag_idx`.
                # Then add `current_emissions[:, next_tag_idx]`.
                term_to_sum = log_alpha + self.transitions[:, next_tag_idx].unsqueeze(0) # [batch, num_prev_tags]
                log_alpha_t[:, next_tag_idx] = torch.logsumexp(term_to_sum, dim=1) + current_emissions[:, next_tag_idx]

            # Masking for sentences that have ended before this timestep t.
            # If attention_mask_seq_first[t,b] is 0 (padding), then log_alpha_t[b,:] reverts to log_alpha[b,:].
            active_batches_at_t = attention_mask_seq_first[t, :].bool() # [batch_size]
            log_alpha = torch.where(active_batches_at_t.unsqueeze(1), log_alpha_t, log_alpha)

        # After iterating through all words, add transition to STOP_TAG.
        final_scores = log_alpha + self.transitions[:, self.tag_to_idx[STOP_TAG]].unsqueeze(0)
        total_log_Z = torch.logsumexp(final_scores, dim=1) # [batch_size]

        return total_log_Z


    def calculate_loss(self, sentences, tags, attention_mask):
        """ Calculates the negative log likelihood loss. """
        # sentences: [batch_size, seq_len]
        # tags: [batch_size, seq_len] (true tag indices, PAD_TAG_ID for padding)
        # attention_mask: [batch_size, seq_len] (True for non-pad, False for pad)

        # Transpose for Transformer/CRF standard [seq_len, batch_size]
        sentences_transposed = sentences.transpose(0, 1)
        tags_transposed = tags.transpose(0, 1)
        attention_mask_seq_first = attention_mask.transpose(0, 1)

        emissions = self._get_transformer_emissions(sentences_transposed, attention_mask) # [seq_len, batch_size, num_tags]

        true_path_score = self._calculate_true_path_score(emissions, tags_transposed, attention_mask_seq_first)
        log_partition_function = self._calculate_log_partition_function(emissions, attention_mask_seq_first)

        # Loss = log Z(x) - Score(y_true, x)
        # Average over the batch.
        loss = (log_partition_function - true_path_score).mean()
        return loss

    def _viterbi_decode(self, emissions, sequence_mask):
        """ Finds the best tag sequence using Viterbi algorithm for a single sentence. """
        # emissions: [seq_len, 1, num_tags] (for a single sentence)
        # sequence_mask: [seq_len, 1] (True for non-pad tokens)
        # Note: This implementation assumes batch_size=1 for simplicity.

        seq_len = emissions.size(0)

        # log_delta[t, j]: max score of a path ending at word t with tag j
        log_delta = torch.full((seq_len, self.num_tags), -10000.0, device=emissions.device)
        # psi[t, j]: backpointer to the previous tag that maximizes score for tag j at word t
        psi = torch.zeros((seq_len, self.num_tags), dtype=torch.long, device=emissions.device)

        # Initialization (t=0)
        # Score(START -> y_0) = emit(y_0, x_0) + trans(START -> y_0)
        initial_scores = self.transitions[self.tag_to_idx[START_TAG], :] + emissions[0, 0, :] # emissions[0,0,:] is for first word, first (only) batch item
        log_delta[0, :] = initial_scores

        # Recursion (t=1 to seq_len-1)
        for t in range(1, seq_len):
            for next_tag_idx in range(self.num_tags):
                # Scores from previous step + transition to current tag: log_delta[t-1, prev_tag] + transitions[prev_tag, next_tag]
                scores_to_next_tag = log_delta[t-1, :] + self.transitions[:, next_tag_idx]
                best_prev_score, best_prev_tag_idx = torch.max(scores_to_next_tag, dim=0)
                # Add emission score for current word and current tag
                log_delta[t, next_tag_idx] = best_prev_score + emissions[t, 0, next_tag_idx]
                psi[t, next_tag_idx] = best_prev_tag_idx

        # Termination: transition to STOP_TAG
        # log_delta[seq_len-1, prev_tag] + transitions[prev_tag, STOP_TAG]
        final_scores = log_delta[seq_len-1, :] + self.transitions[:, self.tag_to_idx[STOP_TAG]]
        best_path_score, best_last_tag_idx = torch.max(final_scores, dim=0)

        # Backtracking to find the best path
        best_path_indices = [best_last_tag_idx.item()]
        for t in range(seq_len - 1, 0, -1):
            # Get the previous tag from backpointer table for the current best tag
            best_prev_tag_idx = psi[t, best_path_indices[-1]]
            best_path_indices.append(best_prev_tag_idx.item())

        best_path_indices.reverse() # Path was constructed backwards
        return best_path_indices, best_path_score

    def predict_tags(self, sentences, attention_mask):
        """ Predicts tag sequences for a batch of sentences. """
        # sentences: [batch_size, seq_len] (token indices)
        # attention_mask: [batch_size, seq_len] (True for non-pad)
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            sentences_transposed = sentences.transpose(0,1) # To [seq_len, batch_size]
            emissions_batch = self._get_transformer_emissions(sentences_transposed, attention_mask) # [seq_len, batch_size, num_tags]

            batch_size = sentences.size(0)
            all_predicted_paths = []

            for i in range(batch_size):
                true_len = attention_mask[i].sum().item() # Actual length of this sentence
                if true_len == 0:
                    all_predicted_paths.append([]) # Empty prediction for empty input
                    continue

                # Slice emissions for the current sentence and its true length
                emissions_single_sentence = emissions_batch[:true_len, i:i+1, :] # [true_len, 1, num_tags]
                # Create a sequence mask for this single sentence (not strictly needed by current _viterbi_decode if true_len is used)
                mask_single_sentence = attention_mask[i:i+1, :true_len].transpose(0,1) # [true_len, 1]

                path_indices, _ = self._viterbi_decode(emissions_single_sentence, mask_single_sentence)
                all_predicted_paths.append(path_indices)
        return all_predicted_paths


# --- Data Reading and Preprocessing ---
def read_conll_data(file_path):
    """ Reads CoNLL-formatted data. Assumes word is first token, tag is last. """
    sentences_words, sentences_tags = [], []
    current_words, current_tags = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences_words.append(current_words)
                    sentences_tags.append(current_tags)
                    current_words, current_tags = [], []
                continue
            parts = line.split()
            current_words.append(parts[0])
            current_tags.append(parts[-1])
    if current_words: # Add last sentence if file doesn't end with blank line
        sentences_words.append(current_words)
        sentences_tags.append(current_tags)
    return sentences_words, sentences_tags

def build_vocab_tags(sentences_words, sentences_tags):
    """ Builds word_to_idx and tag_to_idx mappings from the data. """
    word_counts = defaultdict(int)
    for sentence in sentences_words:
        for word in sentence:
            word_counts[word] += 1

    # word_to_idx: PAD_TOKEN is 0, UNK_TOKEN is 1.
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in word_counts.items():
        if word not in word_to_idx: # Avoid re-assigning PAD/UNK
            word_to_idx[word] = len(word_to_idx)

    # tag_to_idx: Ensure START and STOP tags are included for CRF.
    tag_set = set([tag for sent_tags in sentences_tags for tag in sent_tags])
    tag_to_idx = {tag: i for i, tag in enumerate(list(tag_set))}

    if START_TAG not in tag_to_idx:
        tag_to_idx[START_TAG] = len(tag_to_idx)
    if STOP_TAG not in tag_to_idx:
        tag_to_idx[STOP_TAG] = len(tag_to_idx)

    idx_to_tag = {i: t for t, i in tag_to_idx.items()}
    return word_to_idx, tag_to_idx, idx_to_tag

class NERDataset(Dataset):
    def __init__(self, sentences_words, sentences_tags, word_to_idx, tag_to_idx, max_seq_len):
        self.sentences_words = sentences_words
        self.sentences_tags = sentences_tags # Can be None or dummy for prediction
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sentences_words)

    def __getitem__(self, idx):
        words = self.sentences_words[idx]
        word_indices = [self.word_to_idx.get(w, self.word_to_idx[UNK_TOKEN]) for w in words]
        word_indices = word_indices[:self.max_seq_len] # Truncate
        seq_actual_len = len(word_indices) # Length after potential truncation

        if self.sentences_tags: # During training/evaluation with gold tags
            tags = self.sentences_tags[idx]
            # All training tags should be in tag_to_idx. If not, could indicate an issue.
            tag_indices = [self.tag_to_idx[t] for t in tags] # Map tags to indices
            tag_indices = tag_indices[:self.max_seq_len] # Truncate tags consistently
        else: # During prediction, no gold tags available
            tag_indices = [PAD_TAG_ID] * seq_actual_len # Dummy tags, won't be used for loss

        return torch.tensor(word_indices), torch.tensor(tag_indices, dtype=torch.long), seq_actual_len

def collate_fn(batch, pad_token_idx, pad_tag_id, max_seq_len):
    """ Collates batch data: pads sequences to max_seq_len and creates attention mask. """
    # batch: list of (word_indices_tensor, tag_indices_tensor, seq_actual_len_int)

    words_batch = [item[0] for item in batch]
    tags_batch = [item[1] for item in batch]
    # lengths_batch = [item[2] for item in batch] # Actual lengths, not directly used for padding here but good to have

    # Pad word sequences to max_seq_len
    padded_words_list = []
    for word_seq in words_batch:
        pad_len = max_seq_len - len(word_seq)
        padded_seq = torch.cat([word_seq, torch.full((pad_len,), pad_token_idx, dtype=torch.long)])
        padded_words_list.append(padded_seq)
    padded_words = torch.stack(padded_words_list, dim=0) # [batch_size, max_seq_len]

    # Pad tag sequences to max_seq_len using pad_tag_id
    padded_tags_list = []
    for tag_seq in tags_batch:
        pad_len = max_seq_len - len(tag_seq)
        # If original tag_seq was shorter than max_seq_len due to truncation,
        # it will be padded here. If it was already full (due to dummy tags in prediction),
        # pad_len will be 0.
        padded_seq = torch.cat([tag_seq, torch.full((pad_len,), pad_tag_id, dtype=torch.long)])
        padded_tags_list.append(padded_seq)
    padded_tags = torch.stack(padded_tags_list, dim=0) # [batch_size, max_seq_len]

    # Create attention mask (True for non-pad tokens, False for pad tokens)
    attention_mask = (padded_words != pad_token_idx)

    return padded_words, padded_tags, attention_mask


# --- Training Function ---
def train_model(input_file, model_output_path, num_epochs, learning_rate):
    print(f"Starting Transformer-CRF training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sentences_words, sentences_tags = read_conll_data(input_file)
    actual_max_len = max(len(s) for s in sentences_words if s) if sentences_words else 0
    if actual_max_len == 0:
        print("Warning: No words found in training data or all sentences are empty. Using default max_seq_len.")
        current_max_seq_len = DEFAULT_MAX_SEQ_LEN
    else:
        current_max_seq_len = _get_next_power_of_2(actual_max_len)
    print(f"Actual max sentence length in training data: {actual_max_len}")
    print(f"Using MAX_SEQ_LEN (rounded to power of 2): {current_max_seq_len}")

    wandb.init(project="transformer-crf-ner", config={
        "learning_rate": learning_rate, "epochs": num_epochs,
        "calculated_max_seq_len": current_max_seq_len,
        "embedding_dim": EMBEDDING_DIM, "transformer_hidden_dim": HIDDEN_DIM,
        "transformer_heads": N_HEADS, "transformer_layers": N_LAYERS,
        "dropout": DROPOUT, "device": str(device)
    })

    word_to_idx, tag_to_idx, idx_to_tag = build_vocab_tags(sentences_words, sentences_tags)
    pad_token_idx_for_collate = word_to_idx[PAD_TOKEN]

    train_dataset = NERDataset(sentences_words, sentences_tags, word_to_idx, tag_to_idx, current_max_seq_len)
    custom_collate_fn = functools.partial(collate_fn,
                                          pad_token_idx=pad_token_idx_for_collate,
                                          pad_tag_id=PAD_TAG_ID,
                                          max_seq_len=current_max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    model = TransformerCRF(
        vocab_size=len(word_to_idx),
        tag_to_idx=tag_to_idx,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=word_to_idx[PAD_TOKEN],
        current_max_seq_len=current_max_seq_len
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print("Starting training loop...")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_epoch_loss = 0
        for batch_idx, (padded_words, padded_tags, attention_mask) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            padded_words = padded_words.to(device)
            padded_tags = padded_tags.to(device)       # Target tags for loss calculation
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            loss = model.calculate_loss(padded_words, padded_tags, attention_mask)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping to prevent explosion
                optimizer.step()
                total_epoch_loss += loss.item()
                wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1, "batch_idx": batch_idx})
            else:
                print(f"Warning: NaN or Inf loss encountered at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
                wandb.log({"skipped_batch_loss": loss.item() if loss is not None else -1.0, "epoch": epoch+1, "batch_idx": batch_idx})

        avg_epoch_loss = total_epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        tqdm.write(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_average_loss": avg_epoch_loss, "epoch": epoch + 1})

    model_save_content = {
        'state_dict': model.state_dict(),
        'word_to_idx': word_to_idx,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag,
        'max_seq_len': current_max_seq_len, # Save dynamically calculated max_seq_len
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
        'pad_idx_vocab': word_to_idx[PAD_TOKEN]
    }
    torch.save(model_save_content, model_output_path)
    print(f"Training complete. Model saved to {model_output_path}")
    wandb.finish()

# --- Prediction Function ---
def predict_tags_for_file(model_input_path, input_file, output_file):
    print(f"Starting Transformer-CRF prediction...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(model_input_path, map_location=device)
    word_to_idx = checkpoint['word_to_idx']
    tag_to_idx = checkpoint['tag_to_idx']
    idx_to_tag = checkpoint['idx_to_tag']
    max_seq_len_loaded = checkpoint.get('max_seq_len', DEFAULT_MAX_SEQ_LEN)
    print(f"Loaded model with max_seq_len: {max_seq_len_loaded}")

    model = TransformerCRF(
        vocab_size=len(word_to_idx),
        tag_to_idx=tag_to_idx,
        embedding_dim=checkpoint.get('embedding_dim', EMBEDDING_DIM),
        hidden_dim=checkpoint.get('hidden_dim', HIDDEN_DIM),
        n_heads=checkpoint.get('n_heads', N_HEADS),
        n_layers=checkpoint.get('n_layers', N_LAYERS),
        dropout=checkpoint.get('dropout', DROPOUT),
        pad_idx=checkpoint.get('pad_idx_vocab', word_to_idx.get(PAD_TOKEN, 0)),
        current_max_seq_len=max_seq_len_loaded
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Read input data; for prediction, we primarily need words.
    # Tags read here might be dummy or gold, but NERDataset handles it.
    sentences_words, original_tags_if_any = read_conll_data(input_file)

    # Use NERDataset and DataLoader for consistent preprocessing during prediction.
    # Pass None for sentences_tags if input file only has words.
    pred_dataset = NERDataset(sentences_words, None, word_to_idx, tag_to_idx, max_seq_len_loaded)
    custom_collate_fn_pred = functools.partial(collate_fn,
                                               pad_token_idx=word_to_idx.get(PAD_TOKEN, 0),
                                               pad_tag_id=PAD_TAG_ID,
                                               max_seq_len=max_seq_len_loaded)
    pred_dataloader = DataLoader(pred_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn_pred)

    print(f"Predicting tags for {len(sentences_words)} sentences...")
    original_sentence_idx = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for padded_words_batch, _, attention_mask_batch in tqdm(pred_dataloader, desc="Predicting Batches"):
            padded_words_batch = padded_words_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)

            batch_predicted_paths_indices = model.predict_tags(padded_words_batch, attention_mask_batch)

            for i in range(padded_words_batch.size(0)): # Iterate through sentences in the current batch
                if original_sentence_idx >= len(sentences_words):
                    break # All original sentences processed

                current_original_sentence_words = sentences_words[original_sentence_idx]
                num_actual_words_in_original = len(current_original_sentence_words)

                # Predicted indices correspond to the model's input length (potentially truncated/padded)
                predicted_indices_for_this_sentence = batch_predicted_paths_indices[i]

                for word_idx in range(num_actual_words_in_original):
                    word = current_original_sentence_words[word_idx]
                    if word_idx < len(predicted_indices_for_this_sentence):
                        # This word was within the model's processed sequence length (max_seq_len_loaded)
                        tag = idx_to_tag[predicted_indices_for_this_sentence[word_idx]]
                    else:
                        # This word was truncated during preprocessing (beyond max_seq_len_loaded)
                        tag = "O" # Default tag for truncated words
                    f_out.write(f"{word} {tag}\n")
                f_out.write("\n") # Sentence separator

                original_sentence_idx += 1

            if original_sentence_idx >= len(sentences_words):
                 break # All sentences processed, exit outer loop

    print(f"Prediction complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-CRF for Named Entity Recognition (NER)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Transformer-CRF model")
    train_parser.add_argument("--input", required=True, help="Path to the training data file (CoNLL format)")
    train_parser.add_argument("--model", required=True, help="Path to save the trained model (.pth)")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW optimizer")

    predict_parser = subparsers.add_parser("predict", help="Predict tags using a trained Transformer-CRF model")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model file (.pth)")
    predict_parser.add_argument("--input", required=True, help="Path to the input file for prediction (CoNLL or words-per-line format)")
    predict_parser.add_argument("--output", required=True, help="Path to save the prediction results")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.input, args.model, args.epochs, args.lr)
    elif args.command == "predict":
        predict_tags_for_file(args.model, args.input, args.output) 
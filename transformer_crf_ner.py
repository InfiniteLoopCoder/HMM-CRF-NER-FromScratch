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
import functools # For functools.partial in DataLoader

# Define special tokens and tags
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TAG = "<START>" # For CRF
STOP_TAG = "<STOP>"   # For CRF
PAD_TAG_ID = -1 # Special ID for padding in tags, to be ignored in loss

# Hyperparameters (can be tuned or moved to args)
DEFAULT_MAX_SEQ_LEN = 128  # Default Max sentence length if not calculated
EMBEDDING_DIM = 256
HIDDEN_DIM = 512 # For Transformer's d_model
N_HEADS = 8
N_LAYERS = 3
DROPOUT = 0.1

# Helper function to get the next power of 2
def _get_next_power_of_2(n):
    if n <= 0:
        return 1 # Or handle error appropriately
    power = 1
    while power < n:
        power *= 2
    return power

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerCRF(nn.Module):
    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim, n_heads, n_layers, dropout, pad_idx, current_max_seq_len):
        super(TransformerCRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.num_tags = len(tag_to_idx)
        self.pad_idx = pad_idx # Index for PAD_TOKEN in vocabulary
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=current_max_seq_len + 5)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.hidden2tag = nn.Linear(embedding_dim, self.num_tags)

        # CRF transition parameters
        # transitions[i, j] is the score of transitioning from tag_i to tag_j
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        # We enforce constraints that we never transition to START_TAG and never transition from STOP_TAG
        # These will be handled by setting their scores to -infinity during forward/viterbi if needed,
        # or by ensuring tag_to_idx has START_TAG and STOP_TAG for indexing.
        # Let's ensure START_TAG and STOP_TAG are in tag_to_idx for indexing transitions
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000.0
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000.0


    def _get_transformer_emissions(self, sentences, attention_mask):
        # sentences shape: [seq_len, batch_size]
        # attention_mask shape: [batch_size, seq_len] (True for non-pad, False for pad)
        
        embedded = self.embedding(sentences) * math.sqrt(self.embedding.embedding_dim) # [seq_len, batch_size, emb_dim]
        pos_encoded = self.pos_encoder(embedded) # [seq_len, batch_size, emb_dim]
        
        # Transformer expects src_key_padding_mask: [batch_size, src_seq_len]
        # where True indicates padding. Our attention_mask is True for non-pad. So invert it.
        src_key_padding_mask = ~attention_mask

        transformer_out = self.transformer_encoder(pos_encoded, src_key_padding_mask=src_key_padding_mask)
        # transformer_out shape: [seq_len, batch_size, emb_dim]
        
        emissions = self.hidden2tag(transformer_out) # [seq_len, batch_size, num_tags]
        return emissions

    def _calculate_true_path_score(self, emissions, tags, attention_mask_seq_first):
        # emissions: [seq_len, batch_size, num_tags]
        # tags: [seq_len, batch_size] (true tag indices)
        # attention_mask_seq_first: [seq_len, batch_size] (True for non-pad)
        
        batch_size = emissions.size(1)
        seq_len = emissions.size(0)
        score = torch.zeros(batch_size, device=emissions.device)

        # Add emission scores
        for i in range(batch_size):
            # Only sum up scores for non-padded tokens
            true_len = attention_mask_seq_first[:, i].sum().int()
            # tags[t,i] is the true tag for word t in sentence i
            # emissions[t,i,tag] is the emission score for word t, sentence i, and tag
            score[i] += emissions[torch.arange(true_len), i, tags[torch.arange(true_len), i]].sum()

        # Add transition scores
        # Need to prepend START_TAG to tags for transitions
        start_tag_tensor = torch.full((1, batch_size), self.tag_to_idx[START_TAG], dtype=torch.long, device=tags.device)
        tags_with_start = torch.cat([start_tag_tensor, tags], dim=0)

        for i in range(batch_size):
            true_len = attention_mask_seq_first[:, i].sum().int()
            # Iterate up to true_len (for emissions) which means true_len+1 transitions (START -> T1 -> ... -> T_L -> STOP)
            for t in range(true_len): # transition from tags_with_start[t] to tags_with_start[t+1]
                score[i] += self.transitions[tags_with_start[t, i], tags_with_start[t+1, i]]
            # Add transition from last true tag to STOP_TAG
            score[i] += self.transitions[tags_with_start[true_len, i], self.tag_to_idx[STOP_TAG]]
            
        return score


    def _calculate_log_partition_function(self, emissions, attention_mask_seq_first):
        # emissions: [seq_len, batch_size, num_tags]
        # attention_mask_seq_first: [seq_len, batch_size] (True for non-pad)
        
        seq_len = emissions.size(0)
        batch_size = emissions.size(1)
        
        # log_alpha[b, j] = log prob of paths ending at current word, batch b, with tag j
        log_alpha = torch.full((batch_size, self.num_tags), -10000.0, device=emissions.device)
        
        # Initialization: from START_TAG to first word's tags
        # Score(START -> y_0) = emission_score(y_0, x_0) + transition_score(START -> y_0)
        # Here, emissions[0] are scores for x_0 for each tag y_0
        log_alpha[:, self.tag_to_idx[START_TAG]] = 0.0 # Start with log(1)=0 at START_TAG

        # Iterate through words in the sequence
        for t in range(seq_len):
            # Only update for active batches at this timestep t
            # This masking logic is tricky with batched forward.
            # A simpler way for CRF forward is to iterate per sentence if batch_size > 1,
            # or ensure the masking correctly nullifies padded parts.
            # For now, let's assume we process full sequences and mask later, or iterate.

            # The emission for word t for all tags: emissions[t] (shape [batch_size, num_tags])
            current_emissions = emissions[t] # [batch_size, num_tags]
            
            # log_alpha_next for this step
            log_alpha_t = torch.full((batch_size, self.num_tags), -10000.0, device=emissions.device)

            for next_tag_idx in range(self.num_tags):
                # Score of emission for word t, tag next_tag_idx
                emit_score = current_emissions[:, next_tag_idx].unsqueeze(1) # [batch_size, 1]
                
                # Score of transition from prev_tag to next_tag_idx
                # self.transitions[:, next_tag_idx] is vector of scores trans TO next_tag_idx from ALL prev_tags
                # Shape: [num_tags]
                trans_score = self.transitions[:, next_tag_idx].unsqueeze(0) # [1, num_tags]
                                
                # log_alpha has scores from previous step for each prev_tag
                # log_alpha shape: [batch_size, num_tags]
                # Broadcast sum: log_alpha (paths to prev_tags) + trans_score (prev_tag -> next_tag)
                # Result: [batch_size, num_tags] where element (b, prev_j) is score of path to prev_j then trans to next_tag_idx
                scores_to_next_tag = log_alpha + trans_score # Broadcasting sum
                
                # Add emission score for current word and next_tag_idx
                # scores_to_next_tag will be [batch_size, num_tags], emit_score is [batch_size, 1]
                # We want to add emit_score to all paths ending at next_tag_idx
                # This should be: prev_alpha + transition + current_emission_for_next_tag
                # log_alpha_t[:, next_tag_idx] = torch.logsumexp(log_alpha + trans_score, dim=1) + emit_score.squeeze(1)
                # Correct:
                # log_alpha_t[:, next_tag_idx] = torch.logsumexp(log_alpha + self.transitions[:, next_tag_idx].unsqueeze(0), dim=1) + current_emissions[:, next_tag_idx]

                # For each next_tag, we sum over all previous tags k:
                # logsumexp_k (alpha_prev[k] + transition[k, next_tag] + emission[next_tag])
                # This can be done as: logsumexp_k (alpha_prev[k] + transition[k, next_tag]) + emission[next_tag]
                
                # log_alpha_t[:, next_tag_idx] = torch.logsumexp(log_alpha + self.transitions[:, next_tag_idx].view(1, -1), dim=1) + emissions[t, :, next_tag_idx]
                # This is still slightly off. Let's re-evaluate the loop structure for clarity.
                # log_alpha_prev should be for previous time step
                
                # Let log_dp be the DP table, log_dp[i,j] for word i, tag j.
                # log_dp[i, j] = emissions[i,j] + logsumexp_k (log_dp[i-1, k] + transitions[k,j])

                # Rewriting the forward pass more clearly:
                # Initialize: log_alpha is scores for paths ending at previous word (t-1)
                #             current_emissions is for current word (t)

                # For paths ending at word t with tag `next_tag_idx`:
                # Sum over all `prev_tag_idx`
                #   `log_alpha[:, prev_tag_idx]` # score of path ending at t-1 with `prev_tag_idx`
                # + `self.transitions[prev_tag_idx, next_tag_idx]` # score of transition
                # Then logsumexp over `prev_tag_idx`
                # Then add `current_emissions[:, next_tag_idx]`
                
                # Efficiently:
                #   `log_alpha` : [batch, num_prev_tags]
                #   `self.transitions[:, next_tag_idx]` : [num_prev_tags] -> `view(1, -1)` for broadcasting
                #   `log_alpha + self.transitions[:, next_tag_idx].view(1, -1)` -> `[batch, num_prev_tags]`
                #   `torch.logsumexp(..., dim=1)` -> `[batch]` (Scalar score for each batch item to reach `next_tag_idx`)
                #   `+ current_emissions[:, next_tag_idx]` -> `[batch]`
                
                term_to_sum = log_alpha + self.transitions[:, next_tag_idx].unsqueeze(0) # [batch_size, num_tags(prev)]
                log_alpha_t[:, next_tag_idx] = torch.logsumexp(term_to_sum, dim=1) + current_emissions[:, next_tag_idx]

            # Apply mask for sentences that ended before this time step t
            # `attention_mask_seq_first[t, b]` is 1 if token t for batch b is active, 0 if padding.
            # If `attention_mask_seq_first[t,b]` is 0, then `log_alpha_t[b,:]` should revert to `log_alpha[b,:]`
            # as no new step was taken.
            active_batches_at_t = attention_mask_seq_first[t, :].bool() # [batch_size]
            log_alpha = torch.where(active_batches_at_t.unsqueeze(1), log_alpha_t, log_alpha)


        # After loop, add transition to STOP_TAG
        final_scores = log_alpha + self.transitions[:, self.tag_to_idx[STOP_TAG]].unsqueeze(0)
        total_log_Z = torch.logsumexp(final_scores, dim=1) # [batch_size]
        
        return total_log_Z


    def calculate_loss(self, sentences, tags, attention_mask):
        # sentences: [batch_size, seq_len]
        # tags: [batch_size, seq_len] (true tag indices, -1 for padding)
        # attention_mask: [batch_size, seq_len] (True for non-pad, False for pad)
        
        # Transpose for Transformer/CRF standard [seq_len, batch_size]
        sentences = sentences.transpose(0, 1) # [seq_len, batch_size]
        tags = tags.transpose(0, 1)           # [seq_len, batch_size]
        attention_mask_seq_first = attention_mask.transpose(0, 1) # [seq_len, batch_size]

        emissions = self._get_transformer_emissions(sentences, attention_mask) # [seq_len, batch_size, num_tags]
        
        true_path_score = self._calculate_true_path_score(emissions, tags, attention_mask_seq_first)
        log_partition_function = self._calculate_log_partition_function(emissions, attention_mask_seq_first)
        
        # Negative log likelihood: log Z(x) - Score(y_true, x)
        # We want to average over the batch.
        loss = (log_partition_function - true_path_score).mean()
        return loss

    def _viterbi_decode(self, emissions, sequence_mask):
        # emissions: [seq_len, batch_size, num_tags]
        # sequence_mask: [seq_len, batch_size] (True for non-pad tokens)
        # This Viterbi should process one sentence at a time (batch_size=1 for simplicity in handwritten)
        # or be adapted for batching. For now, let's assume batch_size=1 if called from predict.
        # If used in batch during training (e.g. for metrics), it needs batching.
        
        if emissions.size(1) != 1:
            # print("Warning: Viterbi decode called with batch_size > 1. Processing first sentence only.")
            # Or raise error, or implement batched Viterbi
            # For now, let's assume this will be handled by iterating in predict.
             emissions = emissions[:, 0:1, :] # Take first sentence
             sequence_mask = sequence_mask[:, 0:1]

        seq_len = emissions.size(0)
        
        log_delta = torch.full((seq_len, self.num_tags), -10000.0, device=emissions.device)
        psi = torch.zeros((seq_len, self.num_tags), dtype=torch.long, device=emissions.device)

        # Initialization
        # Score(START -> y_0) = emit(y_0, x_0) + trans(START -> y_0)
        # emissions[0, 0, tag_idx] assumes batch_size=1 (idx 0)
        initial_scores = self.transitions[self.tag_to_idx[START_TAG], :] + emissions[0, 0, :]
        log_delta[0, :] = initial_scores

        # Recursion
        for t in range(1, seq_len):
            for next_tag_idx in range(self.num_tags):
                # log_delta[t-1, prev_tag_idx] + transitions[prev_tag_idx, next_tag_idx]
                scores_to_next_tag = log_delta[t-1, :] + self.transitions[:, next_tag_idx]
                best_prev_score, best_prev_tag_idx = torch.max(scores_to_next_tag, dim=0)
                log_delta[t, next_tag_idx] = best_prev_score + emissions[t, 0, next_tag_idx]
                psi[t, next_tag_idx] = best_prev_tag_idx
        
        # Termination
        # log_delta[seq_len-1, prev_tag_idx] + transitions[prev_tag_idx, STOP_TAG]
        final_scores = log_delta[seq_len-1, :] + self.transitions[:, self.tag_to_idx[STOP_TAG]]
        best_path_score, best_last_tag_idx = torch.max(final_scores, dim=0)
        
        # Backtracking
        best_path = [best_last_tag_idx.item()]
        for t in range(seq_len - 1, 0, -1):
            best_prev_tag_idx = psi[t, best_path[-1]]
            best_path.append(best_prev_tag_idx.item())
        
        best_path.reverse()
        return best_path, best_path_score

    def predict_tags(self, sentences, attention_mask):
        # sentences: [batch_size, seq_len] (token indices)
        # attention_mask: [batch_size, seq_len] (True for non-pad)
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            sentences_transposed = sentences.transpose(0,1)
            emissions = self._get_transformer_emissions(sentences_transposed, attention_mask) # [seq_len, batch_size, num_tags]
            
            batch_size = sentences.size(0)
            all_predicted_paths = []

            for i in range(batch_size):
                # Get actual length of this sentence to only decode that part
                true_len = attention_mask[i].sum().item()
                if true_len == 0:
                    all_predicted_paths.append([])
                    continue

                emissions_single = emissions[:true_len, i:i+1, :] # [true_len, 1, num_tags]
                mask_single = attention_mask[i:i+1, :true_len].transpose(0,1) # [true_len, 1]
                
                path_indices, _ = self._viterbi_decode(emissions_single, mask_single)
                all_predicted_paths.append(path_indices)
        return all_predicted_paths


# --- Data Reading and Preprocessing ---
def read_conll_data(file_path):
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
            current_tags.append(parts[-1]) # Assume tag is the last column
    if current_words: # Add last sentence if file doesn't end with blank line
        sentences_words.append(current_words)
        sentences_tags.append(current_tags)
    return sentences_words, sentences_tags

def build_vocab_tags(sentences_words, sentences_tags, lang='en'): # lang can be used later
    word_counts = defaultdict(int)
    for sentence in sentences_words:
        for word in sentence:
            word_counts[word] += 1
    
    # Create word_to_idx: PAD_TOKEN is 0, UNK_TOKEN is 1
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in word_counts.items(): # Add other words
        # Can add a min_freq threshold here if needed
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    # Create tag_to_idx, ensuring START and STOP tags are present
    tag_set = set([tag for sent_tags in sentences_tags for tag in sent_tags])
    tag_to_idx = {tag: i for i, tag in enumerate(list(tag_set))}
    
    # Add START and STOP tags if not present, crucial for CRF layer
    if START_TAG not in tag_to_idx:
        tag_to_idx[START_TAG] = len(tag_to_idx)
    if STOP_TAG not in tag_to_idx:
        tag_to_idx[STOP_TAG] = len(tag_to_idx)
    
    # PAD_TAG_ID is special and not part of tag_to_idx mapping for model's output layer
    # It's used for targets in DataLoader.
    
    idx_to_tag = {i: t for t, i in tag_to_idx.items()}
    return word_to_idx, tag_to_idx, idx_to_tag

class NERDataset(Dataset):
    def __init__(self, sentences_words, sentences_tags, word_to_idx, tag_to_idx, max_seq_len):
        self.sentences_words = sentences_words
        self.sentences_tags = sentences_tags
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sentences_words)

    def __getitem__(self, idx):
        words = self.sentences_words[idx]
        tags = self.sentences_tags[idx]

        word_indices = [self.word_to_idx.get(w, self.word_to_idx[UNK_TOKEN]) for w in words]
        tag_indices = [self.tag_to_idx.get(t) for t in tags] # Assume all tags are in tag_to_idx from training

        # Truncate
        word_indices = word_indices[:self.max_seq_len]
        tag_indices = tag_indices[:self.max_seq_len]
        
        # Actual length before padding
        seq_actual_len = len(word_indices)

        return torch.tensor(word_indices), torch.tensor(tag_indices), seq_actual_len

def collate_fn(batch, pad_token_idx, pad_tag_id, max_seq_len):
    # batch is a list of tuples (word_indices_tensor, tag_indices_tensor, seq_actual_len_int)
    
    words_batch = [item[0] for item in batch]
    tags_batch = [item[1] for item in batch]
    lengths_batch = [item[2] for item in batch] # Actual lengths

    # Pad sequences
    padded_words = pad_sequence(words_batch, batch_first=True, padding_value=pad_token_idx)
    # For tags, nn.utils.rnn.pad_sequence doesn't take pad_tag_id, so pad manually or ensure it's used correctly
    # We need tags to be padded with a value that loss function can ignore, e.g. PAD_TAG_ID = -1
    # pad_sequence will pad with 0 by default.
    # Let's pad tags manually to MAX_SEQ_LEN
    
    padded_tags_list = []
    for tag_seq in tags_batch:
        padded_seq = torch.full((max_seq_len,), pad_tag_id, dtype=torch.long)
        padded_seq[:len(tag_seq)] = tag_seq
        padded_tags_list.append(padded_seq)
    padded_tags = torch.stack(padded_tags_list, dim=0) # [batch_size, max_seq_len]

    # Truncate/Pad words again to ensure all are exactly max_seq_len
    if padded_words.size(1) > max_seq_len:
        padded_words = padded_words[:, :max_seq_len]
    elif padded_words.size(1) < max_seq_len:
        padding = torch.full((padded_words.size(0), max_seq_len - padded_words.size(1)), pad_token_idx, dtype=torch.long)
        padded_words = torch.cat([padded_words, padding], dim=1)

    # Create attention mask (True for non-pad tokens)
    attention_mask = (padded_words != pad_token_idx)
    
    return padded_words, padded_tags, attention_mask


# --- Training Function ---
def train_model(lang, input_file, model_output_path, num_epochs, learning_rate):
    print(f"Starting Transformer-CRF training for lang: {lang}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sentences_words, sentences_tags = read_conll_data(input_file)
    
    # Calculate dynamic MAX_SEQ_LEN
    actual_max_len = 0
    if sentences_words:
        actual_max_len = max(len(s) for s in sentences_words if s) # Ensure s is not empty
    if actual_max_len == 0: # Handle case of empty training data or all empty sentences
        print("Warning: No words found in training data, or all sentences are empty. Using default max_seq_len.")
        current_max_seq_len = DEFAULT_MAX_SEQ_LEN
    else:
        current_max_seq_len = _get_next_power_of_2(actual_max_len)
    print(f"Actual max sentence length in training data: {actual_max_len}")
    print(f"Using MAX_SEQ_LEN (rounded up to power of 2): {current_max_seq_len}")

    wandb.init(project="transformer-crf-ner", config={
        "language": lang, "learning_rate": learning_rate, "epochs": num_epochs,
        "calculated_max_seq_len": current_max_seq_len, # Log calculated max_seq_len
        "embedding_dim": EMBEDDING_DIM,
        "transformer_hidden_dim": HIDDEN_DIM, "transformer_heads": N_HEADS,
        "transformer_layers": N_LAYERS, "dropout": DROPOUT, "device": str(device)
    })

    word_to_idx, tag_to_idx, idx_to_tag = build_vocab_tags(sentences_words, sentences_tags, lang)
    
    pad_token_idx_for_collate = word_to_idx[PAD_TOKEN]

    train_dataset = NERDataset(sentences_words, sentences_tags, word_to_idx, tag_to_idx, current_max_seq_len)
    
    # Use functools.partial to pass current_max_seq_len to collate_fn
    custom_collate_fn = functools.partial(collate_fn, 
                                          pad_token_idx=pad_token_idx_for_collate, 
                                          pad_tag_id=PAD_TAG_ID, 
                                          max_seq_len=current_max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                                  collate_fn=custom_collate_fn)

    model = TransformerCRF(
        vocab_size=len(word_to_idx),
        tag_to_idx=tag_to_idx,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=word_to_idx[PAD_TOKEN],
        current_max_seq_len=current_max_seq_len # Pass to model for PositionalEncoding
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print("Starting training loop...")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_epoch_loss = 0
        for batch_idx, (padded_words, padded_tags, attention_mask) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            padded_words = padded_words.to(device)
            padded_tags = padded_tags.to(device) # Targets for loss
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            loss = model.calculate_loss(padded_words, padded_tags, attention_mask)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                total_epoch_loss += loss.item()
                wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1, "batch_idx": batch_idx})
            else:
                print(f"Warning: NaN or Inf loss encountered at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
                wandb.log({"skipped_batch_loss": loss.item() if loss is not None else -1.0, "epoch": epoch+1, "batch_idx": batch_idx})


        avg_epoch_loss = total_epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        tqdm.write(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_average_loss": avg_epoch_loss, "epoch": epoch + 1})

    # Save model
    model_save_content = {
        'state_dict': model.state_dict(),
        'word_to_idx': word_to_idx,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag,
        'max_seq_len': current_max_seq_len, # Save the dynamically calculated max_seq_len
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
def predict_model(lang, model_input_path, input_file, output_file):
    print(f"Starting Transformer-CRF prediction for lang: {lang}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(model_input_path, map_location=device)
    word_to_idx = checkpoint['word_to_idx']
    tag_to_idx = checkpoint['tag_to_idx']
    idx_to_tag = checkpoint['idx_to_tag']
    # Load max_seq_len from checkpoint, fallback to default if not present
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
        pad_idx=checkpoint.get('pad_idx_vocab', word_to_idx.get(PAD_TOKEN, 0)), # robust pad_idx loading
        current_max_seq_len=max_seq_len_loaded # Pass to model for PositionalEncoding
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    sentences_words, _ = read_conll_data(input_file) # We only need words for prediction
    
    # Create a dummy NERDataset and DataLoader for prediction for consistent preprocessing
    # No tags needed for dataset, pass None or dummy
    dummy_tags = [[START_TAG] * len(s) for s in sentences_words] # Create dummy tags of same length as words
    
    pred_dataset = NERDataset(sentences_words, dummy_tags, word_to_idx, tag_to_idx, max_seq_len_loaded)
    
    # In predict_model, when setting up pred_dataloader:
    custom_collate_fn_pred = functools.partial(collate_fn, 
                                               pad_token_idx=word_to_idx.get(PAD_TOKEN, 0),
                                               pad_tag_id=PAD_TAG_ID, 
                                               max_seq_len=max_seq_len_loaded)
    pred_dataloader = DataLoader(pred_dataset, batch_size=16, shuffle=False,
                                 collate_fn=custom_collate_fn_pred)


    print(f"Predicting tags for {len(sentences_words)} sentences...")
    # all_sentence_outputs = [] # Store (original_words_list, predicted_tags_list) - No longer needed like this

    original_sentence_idx_for_data_access = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # The pred_dataloader processes data in batches, but we need to map back to original sentences sequentially.
        # We will iterate through the dataloader, get predictions for a batch,
        # then iterate through the original sentences that correspond to that batch.

        for padded_words_batch, _, attention_mask_batch in tqdm(pred_dataloader, desc="Predicting Batches"):
            padded_words_batch = padded_words_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)

            batch_predicted_paths_indices = model.predict_tags(padded_words_batch, attention_mask_batch)
            
            # Process each sentence that was in this batch
            for i in range(padded_words_batch.size(0)): # Iterate through items in the current batch
                if original_sentence_idx_for_data_access >= len(sentences_words):
                    break # All original sentences processed
                
                original_current_sentence_words = sentences_words[original_sentence_idx_for_data_access]
                num_original_words = len(original_current_sentence_words)
                
                # Get the predicted tag indices for the part of the sentence that was fed to the model
                # attention_mask_batch[i] corresponds to the i-th sentence in the batch
                # The number of non-padded tokens for this sentence in the batch:
                processed_len_in_batch = attention_mask_batch[i].sum().item()
                
                predicted_indices_for_model_input = batch_predicted_paths_indices[i][:processed_len_in_batch]
                predicted_tags_for_model_input = [idx_to_tag[idx] for idx in predicted_indices_for_model_input]

                for word_idx in range(num_original_words):
                    word = original_current_sentence_words[word_idx]
                    if word_idx < len(predicted_tags_for_model_input):
                        # This word was part of the (potentially truncated) input to the model
                        tag = predicted_tags_for_model_input[word_idx]
                    else:
                        # This word was beyond max_seq_len_loaded and was truncated
                        tag = "O" # Default tag for truncated words
                    f_out.write(f"{word} {tag}\n")
                f_out.write("\n") # Sentence separator
                
                original_sentence_idx_for_data_access += 1
            
            if original_sentence_idx_for_data_access >= len(sentences_words):
                 break # Break outer loop if all original sentences are done

    print(f"Prediction complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-CRF for NER")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the Transformer-CRF model")
    train_parser.add_argument("--lang", choices=['en', 'zh'], required=True, help="Language (en/zh)")
    train_parser.add_argument("--input", required=True, help="Path to the training data file (CoNLL format)")
    train_parser.add_argument("--model", required=True, help="Path to save the trained model (.pth)")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)") # Reduced default for faster testing
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")


    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict tags using a trained Transformer-CRF model")
    predict_parser.add_argument("--lang", choices=['en', 'zh'], required=True, help="Language (en/zh)")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model file (.pth)")
    predict_parser.add_argument("--input", required=True, help="Path to the input file for prediction (CoNLL or words per line)")
    predict_parser.add_argument("--output", required=True, help="Path to save the prediction results")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.lang, args.input, args.model, args.epochs, args.lr)
    elif args.command == "predict":
        predict_model(args.lang, args.model, args.input, args.output) 
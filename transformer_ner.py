import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os

import wandb
WANDB_AVAILABLE = True

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD_TAG = '<PAD>'  # Use a special PAD tag instead of 'O'

class NERDataset(Dataset):
    """Dataset class for NER data"""
    def __init__(self, sentences, tags, word_to_idx, tag_to_idx):
        self.sentences = sentences
        self.tags = tags
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx] if self.tags else None
        
        # Convert words to indices
        word_indices = []
        for word in sentence:
            if word in self.word_to_idx:
                word_indices.append(self.word_to_idx[word])
            else:
                word_indices.append(self.word_to_idx[UNK_TOKEN])
        
        if tags and tags[0] is not None:  # Check if tags exist and are not None
            tag_indices = [self.tag_to_idx[tag] for tag in tags]
            return torch.tensor(word_indices), torch.tensor(tag_indices)
        else:
            return torch.tensor(word_indices), None

def collate_fn(batch):
    """Custom collate function for padding"""
    sentences, tags = zip(*batch)
    
    # Pad sentences
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    
    if tags[0] is not None:
        # Pad tags
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
        return sentences_padded, tags_padded
    else:
        return sentences_padded, None

class TransformerCRF(nn.Module):
    """Transformer + CRF model for NER"""
    def __init__(self, vocab_size, tag_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, num_heads=4, dropout=0.1, max_seq_len=512):
        super(TransformerCRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear layer to project to tag space
        self.hidden2tag = nn.Linear(embedding_dim, tag_size)
        
        # CRF transition parameters
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size))
        self.start_transitions = nn.Parameter(torch.randn(tag_size))
        self.end_transitions = nn.Parameter(torch.randn(tag_size))
        
        # Initialize transitions
        nn.init.xavier_uniform_(self.transitions)
        nn.init.normal_(self.start_transitions)
        nn.init.normal_(self.end_transitions)
        
    def _get_emission_scores(self, sentences, mask=None):
        """Get emission scores from the transformer"""
        # Embedding
        embeds = self.embedding(sentences)
        embeds = self.pos_encoding(embeds)
        
        # Create attention mask for padding
        if mask is None:
            mask = (sentences != 0)
        
        # Transformer encoding
        transformer_out = self.transformer(embeds, src_key_padding_mask=~mask)
        
        # Project to tag space
        emissions = self.hidden2tag(transformer_out)
        
        return emissions
    
    def forward(self, sentences, tags, mask=None):
        """Forward pass - compute negative log likelihood"""
        if mask is None:
            mask = (sentences != 0)
            
        emissions = self._get_emission_scores(sentences, mask)
        
        # Compute CRF negative log-likelihood
        return self._crf_neg_log_likelihood(emissions, tags, mask)
    
    def _crf_neg_log_likelihood(self, emissions, tags, mask):
        """Compute negative log-likelihood for CRF"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Compute forward score
        forward_score = self._forward_algorithm(emissions, mask)
        
        # Compute gold score
        gold_score = self._score_sentence(emissions, tags, mask)
        
        # Return negative log-likelihood
        return (forward_score - gold_score).mean()
    
    def _forward_algorithm(self, emissions, mask):
        """Forward algorithm for CRF"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize forward variables
        forward_var = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        
        for t in range(1, seq_len):
            # Broadcast forward_var for all next tags
            forward_var_broadcast = forward_var.unsqueeze(2)
            
            # Broadcast transitions for batch
            transitions_broadcast = self.transitions.unsqueeze(0)
            
            # Compute next forward variables
            next_forward_var = forward_var_broadcast + transitions_broadcast + emissions[:, t].unsqueeze(1)
            
            # LogSumExp
            next_forward_var = torch.logsumexp(next_forward_var, dim=1)
            
            # Mask out padding positions
            forward_var = torch.where(mask[:, t].unsqueeze(1), next_forward_var, forward_var)
        
        # Add end transitions
        terminal_var = forward_var + self.end_transitions.unsqueeze(0)
        
        # Get final scores
        final_scores = torch.logsumexp(terminal_var, dim=1)
        
        return final_scores
    
    def _score_sentence(self, emissions, tags, mask):
        """Score a tagged sentence"""
        batch_size, seq_len = tags.shape
        
        # Start transition score
        score = self.start_transitions[tags[:, 0]] + emissions[torch.arange(batch_size), 0, tags[:, 0]]
        
        for t in range(1, seq_len):
            # Transition score
            trans_score = self.transitions[tags[:, t-1], tags[:, t]]
            
            # Emission score
            emit_score = emissions[torch.arange(batch_size), t, tags[:, t]]
            
            # Update score with masking
            score = score + torch.where(mask[:, t], trans_score + emit_score, torch.zeros_like(score))
        
        # Find last valid position for each sequence
        seq_lens = mask.sum(dim=1)
        last_tags = tags[torch.arange(batch_size), seq_lens - 1]
        
        # Add end transition
        score = score + self.end_transitions[last_tags]
        
        return score
    
    def predict(self, sentences, mask=None):
        """Predict tags using Viterbi decoding"""
        if mask is None:
            mask = (sentences != 0)
            
        emissions = self._get_emission_scores(sentences, mask)
        
        # Viterbi decode
        return self.viterbi_decode(emissions, mask)
    
    def viterbi_decode(self, emissions, mask):
        """Manual implementation of Viterbi decoding"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize paths
        path_scores = []
        
        for b in range(batch_size):
            seq_emissions = emissions[b]
            seq_mask = mask[b]
            seq_len_actual = seq_mask.sum().item()
            
            # Skip if sequence is empty
            if seq_len_actual == 0:
                path_scores.append([0] * seq_len)
                continue
            
            # Initialize path indices for this sequence
            path_indices = []
            
            # Initialize scores
            init_scores = self.start_transitions + seq_emissions[0]
            path_scores_seq = [init_scores.unsqueeze(0)]
            
            # Forward pass
            for t in range(1, seq_len_actual):
                # Broadcast previous scores
                prev_scores = path_scores_seq[-1][-1].unsqueeze(1)
                
                # Compute scores for all transitions
                scores = prev_scores + self.transitions + seq_emissions[t].unsqueeze(0)
                
                # Find max scores and indices
                max_scores, max_indices = scores.max(dim=0)
                
                path_scores_seq.append(max_scores.unsqueeze(0))
                path_indices.append(max_indices)
            
            # Termination
            final_scores = path_scores_seq[-1][-1] + self.end_transitions
            _, last_tag = final_scores.max(dim=0)
            
            # Backtrack
            best_path = [last_tag.item()]
            for indices in reversed(path_indices):
                last_tag = indices[last_tag]
                best_path.append(last_tag.item())
            
            best_path.reverse()
            
            # Ensure the path has exactly seq_len elements
            if len(best_path) > seq_len:
                # Truncate if too long
                best_path = best_path[:seq_len]
            elif len(best_path) < seq_len:
                # Pad if too short
                best_path.extend([0] * (seq_len - len(best_path)))
            
            path_scores.append(best_path)
        
        return torch.tensor(path_scores)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def read_data(file_path):
    """Read NER data from file"""
    sentences = []
    tags = []
    
    current_sentence = []
    current_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
                continue
            
            parts = line.split()
            if len(parts) == 2:
                word, tag = parts
                current_sentence.append(word)
                current_tags.append(tag)
            elif len(parts) == 1:
                # Test data - only words
                current_sentence.append(parts[0])
                current_tags.append(None)  # Use None for test data
    
    # Don't forget last sentence
    if current_sentence:
        sentences.append(current_sentence)
        tags.append(current_tags)
    
    return sentences, tags

def build_vocab(sentences, tags):
    """Build vocabulary for words and tags"""
    word_counts = defaultdict(int)
    tag_set = set()
    
    for sentence in sentences:
        for word in sentence:
            word_counts[word] += 1
    
    for tag_seq in tags:
        for tag in tag_seq:
            if tag is not None:  # Skip None tags from test data
                tag_set.add(tag)
    
    # Create word vocabulary
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in word_counts.items():
        if count >= 1:  # Include all words
            word_to_idx[word] = len(word_to_idx)
    
    # Create tag vocabulary
    tag_to_idx = {PAD_TAG: 0}
    for tag in sorted(tag_set):
        if tag != PAD_TAG:
            tag_to_idx[tag] = len(tag_to_idx)
    
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    
    return word_to_idx, tag_to_idx, idx_to_tag

def get_max_len_power_of_2(sentences):
    """Get maximum sentence length rounded up to next power of 2"""
    max_len = max(len(s) for s in sentences)
    # Round up to next power of 2
    power = 1
    while power < max_len:
        power *= 2
    return power

def train(input_path, model_path):
    """Train the Transformer-CRF model"""
    print("Starting training...")
    
    # Read data
    sentences, tags = read_data(input_path)
    
    if not sentences:
        print("Error: No training data found!")
        return
    
    # Build vocabulary
    word_to_idx, tag_to_idx, idx_to_tag = build_vocab(sentences, tags)
    
    # Get max sequence length
    max_seq_len = get_max_len_power_of_2(sentences)
    print(f"Max sequence length (padded to power of 2): {max_seq_len}")
    
    # Pad sequences to max_seq_len
    for i in range(len(sentences)):
        pad_len = max_seq_len - len(sentences[i])
        if pad_len > 0:
            sentences[i].extend([PAD_TOKEN] * pad_len)
            tags[i].extend([PAD_TAG] * pad_len)
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-5
    num_epochs = 20
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    
    # Initialize wandb if available
    if WANDB_AVAILABLE:
        # Initialize wandb with project name and config
        run_name = f"transformer-crf-{os.path.basename(input_path).split('.')[0]}"
        wandb.init(
            project="ner-transformer-crf",
            name=run_name,
            config={
                "vocab_size": len(word_to_idx),
                "tag_size": len(tag_to_idx),
                "max_seq_len": max_seq_len,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "num_sentences": len(sentences),
                "tags": list(tag_to_idx.keys()),
            }
        )
    
    # Create dataset and dataloader
    dataset = NERDataset(sentences, tags, word_to_idx, tag_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TransformerCRF(
        vocab_size=len(word_to_idx),
        tag_size=len(tag_to_idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if WANDB_AVAILABLE:
        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params
        })
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop with tqdm
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        batch_losses = []
        
        # Batch loop with tqdm
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=1, leave=False)
        for batch_idx, (batch_sentences, batch_tags) in enumerate(batch_pbar):
            batch_sentences = batch_sentences.to(device)
            batch_tags = batch_tags.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(batch_sentences, batch_tags)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            # Update parameters
            optimizer.step()
            
            # Record loss
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            
            # Log to wandb every 10 batches
            if WANDB_AVAILABLE and batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": batch_loss,
                    "batch": epoch * len(dataloader) + batch_idx
                })
        
        avg_loss = total_loss / len(dataloader)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        # Log epoch metrics to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model and vocabularies
    model_data = {
        'model_state_dict': model.state_dict(),
        'word_to_idx': word_to_idx,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag,
        'max_seq_len': max_seq_len,
        'model_config': {
            'vocab_size': len(word_to_idx),
            'tag_size': len(tag_to_idx),
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'max_seq_len': max_seq_len
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Training complete. Model saved to {model_path}")
    
    # Finish wandb run
    if WANDB_AVAILABLE:
        wandb.finish()

def predict(model_path, input_path, output_path):
    """Predict tags using the trained model"""
    print("Running prediction...")
    
    # Load model data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    word_to_idx = model_data['word_to_idx']
    tag_to_idx = model_data['tag_to_idx']
    idx_to_tag = model_data['idx_to_tag']
    max_seq_len = model_data['max_seq_len']
    model_config = model_data['model_config']
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TransformerCRF(**model_config).to(device)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    # Read test data
    print("Reading test data...")
    sentences, _ = read_data(input_path)
    original_sentences = [s[:] for s in sentences]  # Keep original sentences
    print(f"Found {len(sentences)} sentences to predict")
    
    # Process sentences for model input
    processed_sentences = []
    for sentence in sentences:
        if len(sentence) > max_seq_len:
            # Truncate if longer than training max
            processed_sentences.append(sentence[:max_seq_len])
        else:
            # Pad if shorter
            padded = sentence + [PAD_TOKEN] * (max_seq_len - len(sentence))
            processed_sentences.append(padded)
    
    # Create dataset
    dataset = NERDataset(processed_sentences, None, word_to_idx, tag_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Predict with progress bar
    all_predictions = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Predicting", total=len(dataloader))
        for batch_sentences, _ in pbar:
            batch_sentences = batch_sentences.to(device)
            
            # Get predictions
            predictions = model.predict(batch_sentences)
            
            all_predictions.extend(predictions.cpu().numpy())
    
    # Write results with progress bar
    print("Writing results...")
    with open(output_path, 'w', encoding='utf-8') as f:
        pbar = tqdm(enumerate(zip(original_sentences, all_predictions)), 
                   desc="Writing output", 
                   total=len(original_sentences))
        
        for i, (original_sent, pred_indices) in pbar:
            # Output all words from the original sentence
            for j, word in enumerate(original_sent):
                if j < len(pred_indices) and j < max_seq_len:
                    # Within model's prediction range
                    tag_idx = pred_indices[j]
                    tag = idx_to_tag[tag_idx]
                    # Don't output PAD tags as 'O'
                    if tag == PAD_TAG:
                        tag = 'O'
                else:
                    # Beyond model's max length - output 'O'
                    tag = 'O'
                f.write(f"{word} {tag}\n")
            
            # Add blank line between sentences
            if i < len(original_sentences) - 1:
                f.write("\n")
    
    print(f"Prediction complete. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-CRF for NER")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the Transformer-CRF model")
    train_parser.add_argument("--input", required=True, help="Path to the training data file")
    train_parser.add_argument("--model", required=True, help="Path to save the trained model")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict tags for a file")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model file")
    predict_parser.add_argument("--input", required=True, help="Path to the input file for prediction")
    predict_parser.add_argument("--output", required=True, help="Path to save the prediction results")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args.input, args.model)
    elif args.command == "predict":
        predict(args.model, args.input, args.output) 
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import math
import wandb # Added for experiment tracking
from tqdm import tqdm # Added for progress bars

# Define special tags
START_TAG = "<START>"
STOP_TAG = "<STOP>"

# Hardcoded template rules (from template_for_crf.utf8)
# Unigram features
TEMPLATE_RULES = [
    "U00:%x[-2,0]",
    "U01:%x[-1,0]",
    "U02:%x[0,0]",
    "U03:%x[1,0]",
    "U04:%x[2,0]",
    "U05:%x[-2,0]/%x[-1,0]",
    "U06:%x[-1,0]/%x[0,0]",
    "U07:%x[-1,0]/%x[1,0]",
    "U08:%x[0,0]/%x[1,0]",
    "U09:%x[1,0]/%x[2,0]",
    # Bigram features (observation-dependent)
    "B00:%x[-2,0]",
    "B01:%x[-1,0]",
    "B02:%x[0,0]",
    "B03:%x[1,0]",
    "B04:%x[2,0]",
    "B05:%x[-2,0]/%x[-1,0]",
    "B06:%x[-1,0]/%x[0,0]",
    "B07:%x[-1,0]/%x[1,0]",
    "B08:%x[0,0]/%x[1,0]",
    "B09:%x[1,0]/%x[2,0]",
    # Pure transition feature (if you want to ensure it's always considered)
    "B" 
]

# Placeholder for feature extraction
def get_feature_vector(sentence_words, position, current_tag_str, prev_tag_str,
                       template_rules, feature_to_idx, tag_to_idx, training_pass=False):
    """
    Extracts active feature indices for the given context.
    If training_pass is True and a feature is new, it adds it to feature_to_idx.
    Returns a list of active feature indices.
    """
    active_features = []
    
    def get_word_from_spec(spec_part, current_pos, sentence):
        """ Parses %x[row,col] and returns the word. Assumes col=0. """
        try:
            row_offset_str = spec_part.split('[')[1].split(',')[0]
            row_offset = int(row_offset_str)
            abs_idx = current_pos + row_offset
            if 0 <= abs_idx < len(sentence):
                return sentence[abs_idx]
            elif abs_idx < 0:
                return "BOS" # Beginning of sentence token
            else:
                return "EOS" # End of sentence token
        except Exception as e:
            # print(f"Warning: Could not parse word spec '{spec_part}'. Error: {e}")
            return None # Or a special token like "PARSE_ERROR"

    for rule_str in template_rules:
        feature_key = None
        if rule_str == "B": # Pure transition feature e.g. B:prev_O_curr_B-PER
            feature_key = f"B:{prev_tag_str}_{current_tag_str}"
        else:
            rule_parts = rule_str.split(':', 1)
            if len(rule_parts) != 2:
                # print(f"Warning: Invalid rule format '{rule_str}'. Skipping.")
                continue
            
            feature_prefix = rule_parts[0] # e.g., U00, U05, B01
            specs = rule_parts[1]       # e.g., %x[-2,0] or %x[-2,0]/%x[-1,0]
            
            observed_elements = []
            spec_definitions = specs.split('/') # Handles single like %x[0,0] and combined like %x[0,0]/%x[1,0]
            
            valid_observation = True
            for spec_part in spec_definitions:
                word = get_word_from_spec(spec_part, position, sentence_words)
                if word is None: # Parsing error for this part
                    valid_observation = False
                    break
                observed_elements.append(word)
            
            if not valid_observation or not observed_elements:
                # print(f"Warning: Could not derive observation for rule '{rule_str}'. Skipping.")
                continue
            
            observed_value_str = "/".join(observed_elements) # e.g. "wordA" or "wordA/wordB"

            if feature_prefix.startswith('U'): # Unigram feature based on observation and current tag
                # Example: U02:observed_value_str_currenttag
                feature_key = f"{feature_prefix}:{observed_value_str}_{current_tag_str}"
            elif feature_prefix.startswith('B'): # Bigram feature based on observation and prev_tag->current_tag transition
                # Example: B02:observed_value_str_prevtag_currenttag
                feature_key = f"{feature_prefix}:{observed_value_str}_{prev_tag_str}_{current_tag_str}"
            # else:
                # print(f"Warning: Unknown feature prefix '{feature_prefix}' in rule '{rule_str}'. Skipping.")

        if feature_key:
            if feature_key not in feature_to_idx:
                if training_pass:
                    feature_to_idx[feature_key] = len(feature_to_idx)
                else:
                    continue # Ignore unknown features if not in training_pass mode
            active_features.append(feature_to_idx[feature_key])
            
    return active_features


def calculate_score_for_features(active_feature_indices, weights, device):
    """Calculates score given active feature indices and weights."""
    # score is initialized on the same device as weights implicitly if weights is involved, 
    # but explicitly setting device for the zero tensor is safer.
    score = torch.tensor(0.0, dtype=torch.float32, device=device)
    if not active_feature_indices:
        return score
    valid_indices = [idx for idx in active_feature_indices if idx < weights.size(0)]
    if valid_indices:
      # Index tensor can be on CPU, PyTorch handles mixed-device indexing if weights is on GPU
      score = weights[torch.tensor(valid_indices, dtype=torch.long)].sum()
    return score


def sentence_score(sentence_words, sentence_tags_str, weights,
                   template_rules, feature_to_idx, tag_to_idx, device):
    """Calculates the score of a given sentence and its true tag sequence."""
    total_score = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    for i in range(len(sentence_words)):
        current_tag_str = sentence_tags_str[i]
        prev_tag_str = sentence_tags_str[i-1] if i > 0 else START_TAG
        
        active_indices = get_feature_vector(sentence_words, i, current_tag_str, prev_tag_str,
                                           template_rules, feature_to_idx, tag_to_idx,
                                           training_pass=False) 
        
        total_score += calculate_score_for_features(active_indices, weights, device)

    if len(sentence_words) > 0:
        final_prev_tag = sentence_tags_str[-1]
    else: 
        final_prev_tag = START_TAG

    active_indices_stop = get_feature_vector(sentence_words, len(sentence_words), STOP_TAG, final_prev_tag,
                                              template_rules, feature_to_idx, tag_to_idx,
                                              training_pass=False)
    total_score += calculate_score_for_features(active_indices_stop, weights, device)
    
    return total_score

def log_forward_algorithm(sentence_words, weights,
                          template_rules, feature_to_idx, tag_to_idx, idx_to_tag, device):
    """Calculates the log partition function (log Z(x)) using the forward algorithm."""
    num_words = len(sentence_words)
    
    actual_tag_list = [tag for tag in idx_to_tag.values() if tag not in [START_TAG, STOP_TAG]]
    actual_num_tags = len(actual_tag_list)
    
    if num_words == 0:
        # Score of START -> STOP for empty sequence
        active_indices_empty_stop = get_feature_vector([], 0, STOP_TAG, START_TAG,
                                                  template_rules, feature_to_idx, tag_to_idx, training_pass=False)
        return calculate_score_for_features(active_indices_empty_stop, weights, device)

    log_alpha_prev = torch.full((actual_num_tags,), -float('inf'), device=device)

    for j_idx, current_tag_str in enumerate(actual_tag_list):
        active_indices = get_feature_vector(sentence_words, 0, current_tag_str, START_TAG,
                                           template_rules, feature_to_idx, tag_to_idx, training_pass=False)
        log_alpha_prev[j_idx] = calculate_score_for_features(active_indices, weights, device)

    for i in range(1, num_words):
        log_alpha_current = torch.full((actual_num_tags,), -float('inf'), device=device)
        for j_idx, current_tag_str in enumerate(actual_tag_list):
            log_sum_exp_terms = []
            for k_idx, prev_tag_str in enumerate(actual_tag_list):
                active_indices_transition = get_feature_vector(sentence_words, i, current_tag_str, prev_tag_str,
                                                               template_rules, feature_to_idx, tag_to_idx, training_pass=False)
                score_i_j_k = calculate_score_for_features(active_indices_transition, weights, device)
                # Ensure terms are on the correct device before stacking for logsumexp
                log_sum_exp_terms.append(log_alpha_prev[k_idx].to(device) + score_i_j_k)
            
            if log_sum_exp_terms:
                 log_alpha_current[j_idx] = torch.logsumexp(torch.stack(log_sum_exp_terms), dim=0)
            else:
                 log_alpha_current[j_idx] = torch.tensor(-float('inf'), device=device)
        log_alpha_prev = log_alpha_current

    final_log_sum_exp_terms = []
    for k_idx, prev_tag_str in enumerate(actual_tag_list):
        active_indices_stop = get_feature_vector(sentence_words, num_words, STOP_TAG, prev_tag_str,
                                                  template_rules, feature_to_idx, tag_to_idx, training_pass=False)
        score_to_stop = calculate_score_for_features(active_indices_stop, weights, device)
        final_log_sum_exp_terms.append(log_alpha_prev[k_idx].to(device) + score_to_stop)
    
    if not final_log_sum_exp_terms:
        return torch.tensor(-float('inf'), device=device)

    log_Z_x = torch.logsumexp(torch.stack(final_log_sum_exp_terms), dim=0)
    return log_Z_x

# Viterbi does not require gradients, but for consistency with weights device:
def viterbi_decode(sentence_words, weights,
                   template_rules, feature_to_idx, tag_to_idx, idx_to_tag, device):
    """Finds the best tag sequence using the Viterbi algorithm."""
    num_words = len(sentence_words)
    actual_tag_list = [tag for tag in idx_to_tag.values() if tag not in [START_TAG, STOP_TAG]]
    actual_num_tags = len(actual_tag_list)

    if actual_num_tags == 0 : return [], torch.tensor(-float('inf'), device=device)

    # dp_table on device, backpointer_table can stay on CPU
    dp_table = torch.full((num_words, actual_num_tags), -float('inf'), device=device)
    backpointer_table = torch.zeros((num_words, actual_num_tags), dtype=torch.long) # Keep on CPU

    if num_words == 0:
        active_indices_empty_stop = get_feature_vector([], 0, STOP_TAG, START_TAG,
                                                  template_rules, feature_to_idx, tag_to_idx, training_pass=False)
        best_score = calculate_score_for_features(active_indices_empty_stop, weights, device)
        return [], best_score

    for j_idx, current_tag_str in enumerate(actual_tag_list):
        active_indices = get_feature_vector(sentence_words, 0, current_tag_str, START_TAG,
                                           template_rules, feature_to_idx, tag_to_idx, training_pass=False)
        dp_table[0, j_idx] = calculate_score_for_features(active_indices, weights, device)

    for i in range(1, num_words):
        for j_idx, current_tag_str in enumerate(actual_tag_list):
            max_score_for_current_tag = torch.tensor(-float('inf'), device=device)
            best_prev_tag_actual_idx = 0
            for k_idx, prev_tag_str in enumerate(actual_tag_list):
                active_indices_transition = get_feature_vector(sentence_words, i, current_tag_str, prev_tag_str,
                                                               template_rules, feature_to_idx, tag_to_idx, training_pass=False)
                score_i_j_k = calculate_score_for_features(active_indices_transition, weights, device)
                current_path_score = dp_table[i-1, k_idx].to(device) + score_i_j_k
                
                if current_path_score > max_score_for_current_tag:
                    max_score_for_current_tag = current_path_score
                    best_prev_tag_actual_idx = k_idx # This is an index, CPU is fine
            
            dp_table[i, j_idx] = max_score_for_current_tag
            backpointer_table[i, j_idx] = best_prev_tag_actual_idx
            
    max_final_score = torch.tensor(-float('inf'), device=device)
    best_last_tag_actual_idx = 0
    
    for k_idx, prev_tag_str in enumerate(actual_tag_list):
        active_indices_stop = get_feature_vector(sentence_words, num_words, STOP_TAG, prev_tag_str,
                                                  template_rules, feature_to_idx, tag_to_idx, training_pass=False)
        score_to_stop = calculate_score_for_features(active_indices_stop, weights, device)
        current_final_score = dp_table[num_words-1, k_idx].to(device) + score_to_stop
        if current_final_score > max_final_score:
            max_final_score = current_final_score
            best_last_tag_actual_idx = k_idx

    best_path_indices = [best_last_tag_actual_idx]
    current_best_tag_idx_for_bp = best_last_tag_actual_idx
    for i in range(num_words - 1, 0, -1):
        # backpointer_table is on CPU, prev_tag_actual_idx will be a Python int or CPU tensor
        prev_tag_actual_idx = backpointer_table[i, current_best_tag_idx_for_bp]
        best_path_indices.append(prev_tag_actual_idx.item() if torch.is_tensor(prev_tag_actual_idx) else prev_tag_actual_idx)
        current_best_tag_idx_for_bp = prev_tag_actual_idx
    
    best_path_indices.reverse()
    predicted_tags_str = [actual_tag_list[idx] for idx in best_path_indices]
    return predicted_tags_str, max_final_score


def train(input_file, model_output_path, num_epochs=10, learning_rate=0.01):
    print(f"Starting CRF training...")
    print(f"Input: {input_file}, Model Output: {model_output_path}")
    print(f"Epochs: {num_epochs}, LR: {learning_rate}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project="crf-ner", 
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "input_file": input_file,
            "model_output_path": model_output_path,
            "hardcoded_templates_used": True,
            "device": str(device)
        }
    )

    template_rules = TEMPLATE_RULES # Use hardcoded templates
    
    sentences_words = []
    sentences_tags_str = []
    tag_set = set([START_TAG, STOP_TAG]) # Ensure START/STOP are in tag_set from the beginning
    
    current_words = []
    current_tags = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences_words.append(current_words)
                    sentences_tags_str.append(current_tags)
                    current_words = []
                    current_tags = []
                continue
            parts = line.split()
            word, tag = parts[0], parts[-1] # Assuming tag is the last part
            current_words.append(word)
            current_tags.append(tag)
            tag_set.add(tag)
        if current_words: # Add last sentence
            sentences_words.append(current_words)
            sentences_tags_str.append(current_tags)

    tag_to_idx = {tag: i for i, tag in enumerate(list(tag_set))}
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
    
    feature_to_idx = {} 
    # First pass to build feature_to_idx from true paths
    # And to ensure all pure B:prev_curr transitions are there.
    print("Building feature map from training data (true paths)...")
    for i in range(len(sentences_words)):
        sentence_w = sentences_words[i]
        sentence_t_str = sentences_tags_str[i]
        for j in range(len(sentence_w)):
            curr_t = sentence_t_str[j]
            prev_t = sentence_t_str[j-1] if j > 0 else START_TAG
            get_feature_vector(sentence_w, j, curr_t, prev_t, template_rules, feature_to_idx, tag_to_idx, training_pass=True)
        # For STOP_TAG transition
        if sentence_w:
            get_feature_vector(sentence_w, len(sentence_w), STOP_TAG, sentence_t_str[-1], template_rules, feature_to_idx, tag_to_idx, training_pass=True)
        else: # Empty sentence
             get_feature_vector([], 0, STOP_TAG, START_TAG, template_rules, feature_to_idx, tag_to_idx, training_pass=True)


    # Add all pure "B:prev_curr" transitions to feature_to_idx if "B" is in template
    if "B" in template_rules:
        all_tags_for_pure_b = list(tag_set) # Includes START, STOP
        for prev_tag_str in all_tags_for_pure_b:
            for current_tag_str in all_tags_for_pure_b:
                if prev_tag_str == STOP_TAG or current_tag_str == START_TAG: # Invalid transitions
                    continue
                feature_key = f"B:{prev_tag_str}_{current_tag_str}"
                if feature_key not in feature_to_idx:
                     feature_to_idx[feature_key] = len(feature_to_idx)
    
    num_features = len(feature_to_idx)
    print(f"Number of unique features: {num_features}")
    if num_features == 0:
        print("Warning: No features extracted. Check template and data. Aborting training.")
        # Save an empty/minimal model
        model_data = {
            'weights': torch.tensor([]),
            'feature_to_idx': {}, 'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
            'template_rules': template_rules
        }
        with open(model_output_path, 'wb') as f_model:
            pickle.dump(model_data, f_model)
        print(f"Empty model saved to {model_output_path}")
        return

    weights = nn.Parameter(torch.zeros(num_features, dtype=torch.float32, device=device))
    optimizer = optim.SGD([weights], lr=learning_rate)

    print("Starting training loop...")
    # Wrap epoch loop with tqdm
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_epoch_loss = 0.0
        # Wrap sentence loop with tqdm
        sentence_iterator = tqdm(range(len(sentences_words)), desc=f"Epoch {epoch+1}/{num_epochs} Sentences", leave=False)
        for i in sentence_iterator:
            optimizer.zero_grad()
            
            words = sentences_words[i]
            true_tags = sentences_tags_str[i]
            
            # Calculate score of the true path
            s_gold = sentence_score(words, true_tags, weights, template_rules, feature_to_idx, tag_to_idx, device)
            
            # Calculate log partition function
            log_Z = log_forward_algorithm(words, weights, template_rules, feature_to_idx, tag_to_idx, idx_to_tag, device)
            
            # Loss = log Z(x) - Score(y_true, x)
            loss = log_Z - s_gold
            
            # Backpropagate
            if not torch.isinf(loss) and not torch.isnan(loss): # Avoid issues with bad gradients
                loss.backward()
                optimizer.step()
                current_loss_item = loss.item()
                total_epoch_loss += current_loss_item
                # Log sentence loss to wandb
                wandb.log({"sentence_loss": current_loss_item, "sentence_idx": i, "epoch": epoch + 1})
            else:
                current_loss_item = loss.item() # could be inf or nan
                print(f"Warning: Skipping backward pass for sentence {i} due to inf/nan loss: {current_loss_item}")
                wandb.log({"sentence_loss_skipped": current_loss_item, "sentence_idx": i, "epoch": epoch + 1})

            if (i + 1) % 100 == 0:
                # Update tqdm description with current loss if needed, though tqdm shows iteration count
                # sentence_iterator.set_postfix_str(f"Curr Loss: {current_loss_item:.4f}")
                print(f"Epoch {epoch+1}/{num_epochs}, Sentence {i+1}/{len(sentences_words)}, Current Loss: {current_loss_item:.4f}")
        
        avg_epoch_loss = total_epoch_loss / len(sentences_words) if len(sentences_words) > 0 else 0
        # print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}") # tqdm will show epoch completion
        tqdm.write(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}") # Use tqdm.write for cleaner output with progress bars
        # Log epoch average loss to wandb
        wandb.log({"epoch_average_loss": avg_epoch_loss, "epoch": epoch + 1})

    # Save model
    model_data = {
        # Move weights to CPU before saving for portability
        'weights': weights.data.cpu(), 
        'feature_to_idx': feature_to_idx,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag,
    }
    with open(model_output_path, 'wb') as f_model:
        pickle.dump(model_data, f_model)
    print(f"Training complete. Model saved to {model_output_path}")
    wandb.finish() # Finish W&B run


def predict(model_input_path, input_file, output_file):
    print(f"Starting CRF prediction...")
    print(f"Model: {model_input_path}, Input: {input_file}, Output: {output_file}")

    # Determine device for prediction (can be CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for prediction: {device}")

    with open(model_input_path, 'rb') as f_model:
        model_data = pickle.load(f_model)

    # Weights were saved on CPU, move to current device for prediction
    weights = model_data['weights'].to(device)
    feature_to_idx = model_data['feature_to_idx']
    tag_to_idx = model_data['tag_to_idx']
    idx_to_tag = model_data['idx_to_tag']
    # template_rules = model_data['template_rules'] # Use hardcoded
    template_rules = TEMPLATE_RULES

    sentences_words = []
    current_words = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences_words.append(current_words)
                    current_words = []
                continue
            # Assuming input for prediction is just words, one per line
            # or word and some other info, word is the first part.
            parts = line.split()
            if parts:
                 current_words.append(parts[0])
        if current_words:
            sentences_words.append(current_words)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for words in sentences_words:
            if not words:
                f_out.write("\n")
                continue
            
            predicted_tags, _ = viterbi_decode(words, weights, template_rules, feature_to_idx, tag_to_idx, idx_to_tag, device)
            
            for word, tag in zip(words, predicted_tags):
                f_out.write(f"{word} {tag}\n")
            f_out.write("\n")
            
    print(f"Prediction complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear-chain CRF for NER")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the CRF model")
    train_parser.add_argument("--input", required=True, help="Path to the training data file (CoNLL format)")
    train_parser.add_argument("--model", required=True, help="Path to save the trained model")
    train_parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 10)")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")


    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict tags using a trained CRF model")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model file")
    predict_parser.add_argument("--input", required=True, help="Path to the input file for prediction (words per line)")
    predict_parser.add_argument("--output", required=True, help="Path to save the prediction results")

    args = parser.parse_args()

    if args.command == "train":
        train(args.input, args.model, num_epochs=args.epochs, learning_rate=args.lr)
    elif args.command == "predict":
        # Predict function now handles its own device selection
        predict(args.model, args.input, args.output) 
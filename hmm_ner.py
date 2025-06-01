import argparse
import pickle
from collections import defaultdict
import math

# Constant for unknown words/tags, used for smoothing to avoid log(0).
SMOOTHING_FACTOR = 1e-10

def train(input_path, model_path):
    """Trains the HMM model and saves it to a file."""
    print(f"Starting training...")

    word_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    initial_tag_counts = defaultdict(int)
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    
    sentences = []
    current_sentence = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            word, tag = line.split()
            current_sentence.append((word, tag))
            word_counts[word] += 1
            tag_counts[tag] += 1
    if current_sentence: # Add the last sentence if the file doesn't end with a blank line
        sentences.append(current_sentence)

    word_to_idx = {word: i for i, word in enumerate(word_counts.keys())}
    tag_to_idx = {tag: i for i, tag in enumerate(tag_counts.keys())}
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}

    V = len(word_to_idx)
    num_tags = len(tag_to_idx)

    for sentence in sentences:
        if not sentence: continue
        first_word, first_tag = sentence[0]
        initial_tag_counts[first_tag] += 1
        emission_counts[first_tag][first_word] += 1

        for i in range(len(sentence) - 1):
            prev_word, prev_tag = sentence[i]
            curr_word, curr_tag = sentence[i+1]
            
            transition_counts[prev_tag][curr_tag] += 1
            emission_counts[curr_tag][curr_word] += 1

    # Initial probabilities (pi)
    pi = [0.0] * num_tags
    total_sentences = len(sentences)
    if total_sentences == 0 or num_tags == 0 or V == 0:
        print("Error: Training data is empty or does not contain enough information. Cannot train model.")
        # Save a placeholder model or exit
        model_data = {
            'pi': [], 'A': [], 'B': [], 'word_to_idx': {}, 'tag_to_idx': {},
            'idx_to_tag': {}, 'V': 0, 'tag_counts': {}
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Saved an empty/placeholder model to {model_path}")
        return

    for tag, count in initial_tag_counts.items():
        pi[tag_to_idx[tag]] = math.log((count + SMOOTHING_FACTOR) / (total_sentences + num_tags * SMOOTHING_FACTOR))

    # Transition probabilities (A)
    A = [[0.0] * num_tags for _ in range(num_tags)]
    for prev_tag, next_tags in transition_counts.items():
        prev_tag_idx = tag_to_idx[prev_tag]
        total_transitions_from_prev = sum(next_tags.values())
        for next_tag, count in next_tags.items():
            next_tag_idx = tag_to_idx[next_tag]
            A[prev_tag_idx][next_tag_idx] = math.log((count + 1) / (total_transitions_from_prev + num_tags))
        # Handle transitions to tags not seen after prev_tag
        for t_idx in range(num_tags):
            if A[prev_tag_idx][t_idx] == 0.0:
                 A[prev_tag_idx][t_idx] = math.log(1 / (total_transitions_from_prev + num_tags))

    # Emission probabilities (B)
    B = [[0.0] * V for _ in range(num_tags)]
    idx_to_word = {i: w for w, i in word_to_idx.items()} # Create reverse mapping for iteration if needed

    for t_idx in range(num_tags):
        tag_str = idx_to_tag[t_idx]
        current_tag_total_count = tag_counts.get(tag_str, 0) # count(t)

        for w_idx in range(V):
            # word_str = idx_to_word[w_idx] # Not strictly needed if using emission_counts with string keys
            # actual_emission_count = emission_counts[tag_str].get(word_str, 0)
            
            # To get count(w,t), we need word_str for emission_counts
            # emission_counts is defaultdict(lambda: defaultdict(int))
            # emission_counts[tag_string][word_string]
            word_string_for_lookup = idx_to_word[w_idx]
            actual_emission_count = emission_counts[tag_str].get(word_string_for_lookup, 0)

            if current_tag_total_count + V == 0: # Should not happen if V > 0 and tags exist
                B[t_idx][w_idx] = float('-inf') 
            else:
                B[t_idx][w_idx] = math.log((actual_emission_count + 1.0) / (current_tag_total_count + V))

    model_data = {
        'pi': pi,
        'A': A,
        'B': B,
        'word_to_idx': word_to_idx,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag, # Added for convenience in predict/viterbi
        'V': V,
        'tag_counts': {tag_to_idx[tag]: count for tag, count in tag_counts.items()} # Store tag counts for unknown word handling
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Training complete. Model saved to {model_path}")


def predict(model_path, input_path, output_path):
    """Loads a model and predicts tags for a given input file."""
    print(f"Running prediction...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    sentences = []
    current_sentence = []
    # Read test data (words only, no tags)
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            # The input for prediction might just be words, one per line, or word and a dummy tag
            # Assuming it's just words, or words with some placeholder that we ignore.
            parts = line.split() # Robustly handle lines with or without tags
            current_sentence.append(parts[0]) 
    if current_sentence:
        sentences.append(current_sentence)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for sentence_words in sentences:
            if not sentence_words: # Handle potential empty sentences if input has multiple blank lines
                f_out.write("\n")
                continue
            predicted_tags_indices = viterbi_decode(sentence_words, model_data)
            idx_to_tag = model_data['idx_to_tag']
            for word, tag_idx in zip(sentence_words, predicted_tags_indices):
                f_out.write(f"{word} {idx_to_tag[tag_idx]}\n")
            f_out.write("\n") # Sentence separator

    print(f"Prediction complete. Output saved to {output_path}")


def viterbi_decode(sentence_words, model_data):
    """
    Performs Viterbi decoding to find the most likely sequence of tags.
    MUST handle unknown words from the test set.
    """
    pi = model_data['pi']
    A = model_data['A']
    B = model_data['B']
    word_to_idx = model_data['word_to_idx']
    tag_to_idx = model_data['tag_to_idx']
    idx_to_tag = model_data['idx_to_tag'] # For converting indices back to tags if needed for debugging
    V = model_data['V']
    # model_tag_counts is a dict {tag_idx: count}, not {tag_string: count}
    model_tag_counts = model_data['tag_counts'] 

    num_tags = len(tag_to_idx)
    sentence_len = len(sentence_words)

    # dp table: dp[t][j] stores the max probability of any path ending at word t with tag j
    dp = [[float('-inf')] * num_tags for _ in range(sentence_len)]
    # backpointer table: bp[t][j] stores the tag index of word t-1 that leads to max prob for tag j at word t
    bp = [[0] * num_tags for _ in range(sentence_len)]

    # Initialization step (for the first word)
    first_word = sentence_words[0]
    for j in range(num_tags):
        emission_prob_log = float('-inf')
        tag_total_count_for_smoothing = model_tag_counts.get(j, 0) # Get count(t) for tag index j

        if first_word in word_to_idx:
            word_idx = word_to_idx[first_word]
            # B is num_tags x V. word_idx should be < V.
            if j < len(B) and word_idx < len(B[j]):
                 emission_prob_log = B[j][word_idx] # This now has the correctly smoothed P(word|tag)
            else:
                 # This case should ideally not be hit if B is V-dimensional and word_idx is valid.
                 # Fallback to unknown word logic if something is wrong, though B should cover all known words.
                 if tag_total_count_for_smoothing + V == 0:
                    emission_prob_log = float('-inf')
                 else:
                    emission_prob_log = math.log(1.0 / (tag_total_count_for_smoothing + V))
        else: # Unknown word
            if tag_total_count_for_smoothing + V == 0:
                emission_prob_log = float('-inf')
            else:
                emission_prob_log = math.log(1.0 / (tag_total_count_for_smoothing + V))
        
        if not pi: # Handle empty model case
            dp[0][j] = float('-inf')
        else:
            dp[0][j] = pi[j] + emission_prob_log

    # Recursion step
    for t in range(1, sentence_len):
        word = sentence_words[t]
        for j in range(num_tags): # Current tag index
            max_prob = float('-inf')
            best_prev_tag = 0
            tag_total_count_for_smoothing = model_tag_counts.get(j, 0) # Get count(t) for tag index j
            
            emission_prob_log = float('-inf')
            if word in word_to_idx:
                word_idx = word_to_idx[word]
                if j < len(B) and word_idx < len(B[j]): 
                    emission_prob_log = B[j][word_idx]
                else:
                    if tag_total_count_for_smoothing + V == 0:
                        emission_prob_log = float('-inf')
                    else:
                        emission_prob_log = math.log(1.0 / (tag_total_count_for_smoothing + V))
            else: # Unknown word
                 if tag_total_count_for_smoothing + V == 0:
                    emission_prob_log = float('-inf')
                 else:
                    emission_prob_log = math.log(1.0 / (tag_total_count_for_smoothing + V))

            for i in range(num_tags): # Previous tag index
                # Ensure A[i][j] is valid; if it was 0, its log is -inf
                # Check if A[i] has j as a valid index (i.e. A is num_tags x num_tags)
                trans_prob_log = A[i][j] if i < len(A) and j < len(A[i]) else float('-inf')
                
                current_prob = dp[t-1][i] + trans_prob_log + emission_prob_log
                if current_prob > max_prob:
                    max_prob = current_prob
                    best_prev_tag = i
            
            dp[t][j] = max_prob
            bp[t][j] = best_prev_tag

    # Termination step (find the best tag for the last word)
    max_prob_last_word = float('-inf')
    best_last_tag = 0
    for j in range(num_tags):
        if dp[sentence_len-1][j] > max_prob_last_word:
            max_prob_last_word = dp[sentence_len-1][j]
            best_last_tag = j

    # Backtracking
    predicted_tag_indices = [0] * sentence_len
    predicted_tag_indices[sentence_len-1] = best_last_tag
    for t in range(sentence_len-2, -1, -1):
        predicted_tag_indices[t] = bp[t+1][predicted_tag_indices[t+1]]
    
    return predicted_tag_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM for NER")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command parser
    train_parser = subparsers.add_parser("train", help="Train the HMM model")
    train_parser.add_argument("--input", required=True, help="Path to the training data file")
    train_parser.add_argument("--model", required=True, help="Path to save the trained model")

    # Predict command parser
    predict_parser = subparsers.add_parser("predict", help="Predict tags for a file")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model file")
    predict_parser.add_argument("--input", required=True, help="Path to the input file for prediction")
    predict_parser.add_argument("--output", required=True, help="Path to save the prediction results")

    args = parser.parse_args()

    if args.command == "train":
        train(args.input, args.model)
    elif args.command == "predict":
        predict(args.model, args.input, args.output) 
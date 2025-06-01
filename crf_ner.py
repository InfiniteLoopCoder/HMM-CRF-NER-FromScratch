import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import math
import wandb
from tqdm import tqdm

# Define special tags
START_TAG = "<START>"
STOP_TAG = "<STOP>"

# Hardcoded template rules for feature generation
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
    # Pure transition feature
    "B"
]

# --- Optimizations: Pre-parsing Template Rules ---
def parse_template_rule(rule_str):
    """ Parses a single template rule string into a structured dictionary. """
    rule_details = {'original_rule': rule_str, 'prefix': None, 'type': None, 'offsets': []}

    if rule_str == "B":
        rule_details['prefix'] = "B"
        rule_details['type'] = "B_pure" # Pure transition feature
        return rule_details

    parts = rule_str.split(':', 1)
    if len(parts) != 2:
        return None # Malformed rule, skip

    rule_details['prefix'] = parts[0]
    if rule_details['prefix'].startswith('U'):
        rule_details['type'] = 'U'
    elif rule_details['prefix'].startswith('B'):
        rule_details['type'] = 'B_obs' # Observation-dependent Bigram
    else:
        return None # Unknown prefix type

    spec_definitions = parts[1].split('/')
    # Handles rules like U00: (empty spec) which are valid for features like "U00:_TAG"
    if not parts[1] and rule_details['type'] in ['U', 'B_obs']:
        pass
    elif not parts[1] and rule_details['type'] not in ['U', 'B_obs']:
        return None # Invalid spec for other types

    for spec_part in spec_definitions:
        # Handles cases like "U00:" where spec_definitions might be [''] if spec is empty
        if not spec_part:
            if rule_details['type'] in ['U', 'B_obs'] and not parts[1]: # Only if original spec was empty
                continue
            else: # Malformed spec part for rules expecting offsets
                return None

        try:
            # Expecting format like %x[offset,0]
            if not (spec_part.startswith('%x[') and spec_part.endswith(']')):
                return None # Malformed spec part

            offset_str = spec_part.split('[')[1].split(',')[0]
            rule_details['offsets'].append(int(offset_str))
        except (IndexError, ValueError):
            return None # Parsing error for offset
    return rule_details

def pre_parse_all_template_rules(template_rules_list):
    """ Parses all template rule strings and stores them. """
    parsed_rules = []
    for rule_str in template_rules_list:
        parsed = parse_template_rule(rule_str)
        if parsed: # Only add successfully parsed rules
            parsed_rules.append(parsed)
    return parsed_rules

PRE_PARSED_TEMPLATE_RULES = pre_parse_all_template_rules(TEMPLATE_RULES)
# --- End Optimizations ---

# Pre-filter rules for efficiency
U_RULES = [r for r in PRE_PARSED_TEMPLATE_RULES if r and r['type'] == 'U']
B_OBS_RULES = [r for r in PRE_PARSED_TEMPLATE_RULES if r and r['type'] == 'B_obs']
# Pure B rule features are of the form "B:prev_tag_current_tag"
# Check if any rule implies such features might be generated (e.g. by a simple "B" rule string).
HAS_PURE_B_RULE = any(r['type'] == 'B_pure' for r in PRE_PARSED_TEMPLATE_RULES if r)


def get_feature_vector(sentence_words, position, current_tag_str, prev_tag_str,
                       feature_to_idx, tag_to_idx, training_pass=False):
    """
    Extracts active feature indices for the given context using pre-parsed rules.
    If training_pass is True and a feature is new, it adds it to feature_to_idx.
    Returns a list of active feature indices.
    """
    active_features = []
    sentence_len = len(sentence_words)

    def get_word_at_offset(offset, current_pos, words, length):
        """ Safely gets a word from sentence_words based on offset, returning BOS/EOS for out-of-bounds. """
        abs_idx = current_pos + offset
        if 0 <= abs_idx < length:
            return words[abs_idx]
        elif abs_idx < 0:
            return "BOS"
        else:
            return "EOS"

    for rule_details in PRE_PARSED_TEMPLATE_RULES:
        feature_key = None

        if rule_details['type'] == "B_pure":
            feature_key = f"B:{prev_tag_str}_{current_tag_str}"
        else:
            # Unigram ('U') or Observation-dependent Bigram ('B_obs')
            observed_elements = []
            for offset in rule_details['offsets']:
                word = get_word_at_offset(offset, position, sentence_words, sentence_len)
                observed_elements.append(word)

            observed_value_str = "/".join(observed_elements)

            if rule_details['type'] == 'U':
                feature_key = f"{rule_details['prefix']}:{observed_value_str}_{current_tag_str}"
            elif rule_details['type'] == 'B_obs':
                feature_key = f"{rule_details['prefix']}:{observed_value_str}_{prev_tag_str}_{current_tag_str}"

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
    # Explicitly setting device for the zero tensor is safer.
    score = torch.tensor(0.0, dtype=torch.float32, device=device)
    if not active_feature_indices:
        return score
    valid_indices = [idx for idx in active_feature_indices if idx < weights.size(0)]
    if valid_indices:
      # Index tensor can be on CPU; PyTorch handles mixed-device indexing if weights is on GPU.
      score = weights[torch.tensor(valid_indices, dtype=torch.long)].sum()
    return score


def sentence_score(sentence_words, sentence_tags_str, weights,
                   feature_to_idx, tag_to_idx, device):
    """Calculates the score of a given sentence and its true tag sequence."""
    total_score = torch.tensor(0.0, dtype=torch.float32, device=device)
    sentence_len = len(sentence_words)

    def get_word_at_offset_score(offset, current_pos, words, length):
        abs_idx = current_pos + offset
        if 0 <= abs_idx < length: return words[abs_idx]
        elif abs_idx < 0: return "BOS"
        else: return "EOS"

    for i in range(sentence_len):
        current_tag_str = sentence_tags_str[i]
        prev_tag_str = sentence_tags_str[i-1] if i > 0 else START_TAG

        # Unigram features
        for rule_details in U_RULES:
            observed_elements = [get_word_at_offset_score(off, i, sentence_words, sentence_len) for off in rule_details['offsets']]
            observed_value_str = "/".join(observed_elements)
            feature_key = f"{rule_details['prefix']}:{observed_value_str}_{current_tag_str}"
            if feature_key in feature_to_idx:
                total_score += weights[feature_to_idx[feature_key]]

        # Observation-dependent Bigrams
        for rule_details in B_OBS_RULES:
            observed_elements = [get_word_at_offset_score(off, i, sentence_words, sentence_len) for off in rule_details['offsets']]
            observed_value_str = "/".join(observed_elements)
            feature_key = f"{rule_details['prefix']}:{observed_value_str}_{prev_tag_str}_{current_tag_str}"
            if feature_key in feature_to_idx:
                total_score += weights[feature_to_idx[feature_key]]

        # Pure Bigram transitions
        if HAS_PURE_B_RULE:
            feature_key = f"B:{prev_tag_str}_{current_tag_str}"
            if feature_key in feature_to_idx:
                total_score += weights[feature_to_idx[feature_key]]

    # Transition to STOP_TAG
    final_prev_tag = sentence_tags_str[-1] if sentence_len > 0 else START_TAG

    # Unigram features for STOP_TAG (at position sentence_len)
    for rule_details in U_RULES:
        observed_elements = [get_word_at_offset_score(off, sentence_len, sentence_words, sentence_len) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        feature_key = f"{rule_details['prefix']}:{observed_value_str}_{STOP_TAG}"
        if feature_key in feature_to_idx:
            total_score += weights[feature_to_idx[feature_key]]

    # Observation-dependent Bigrams to STOP_TAG
    for rule_details in B_OBS_RULES:
        observed_elements = [get_word_at_offset_score(off, sentence_len, sentence_words, sentence_len) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        feature_key = f"{rule_details['prefix']}:{observed_value_str}_{final_prev_tag}_{STOP_TAG}"
        if feature_key in feature_to_idx:
            total_score += weights[feature_to_idx[feature_key]]

    # Pure Bigram transition to STOP_TAG
    if HAS_PURE_B_RULE:
        feature_key = f"B:{final_prev_tag}_{STOP_TAG}"
        if feature_key in feature_to_idx:
            total_score += weights[feature_to_idx[feature_key]]

    return total_score

def log_forward_algorithm(sentence_words, weights,
                          feature_to_idx, tag_to_idx, idx_to_tag, device):
    """Calculates the log partition function (log Z(x)) using the forward algorithm."""
    num_words = len(sentence_words)

    actual_tag_list_py = [tag for tag in idx_to_tag.values() if tag not in [START_TAG, STOP_TAG]]
    actual_num_tags = len(actual_tag_list_py)

    if num_words == 0:
        # Score of START -> STOP for empty sequence.
        active_indices_empty_stop = get_feature_vector([], 0, STOP_TAG, START_TAG,
                                                  feature_to_idx, tag_to_idx, training_pass=False)
        return calculate_score_for_features(active_indices_empty_stop, weights, device)

    def get_word_at_offset_fwd(offset, current_pos, words, length):
        abs_idx = current_pos + offset
        if 0 <= abs_idx < length: return words[abs_idx]
        elif abs_idx < 0: return "BOS"
        else: return "EOS"

    # log_alpha_prev stores log probabilities for each tag at the previous step
    log_alpha_prev = torch.full((actual_num_tags,), -float('inf'), device=device)

    # --- Initialization step (t=0) ---
    # Calculate scores for START_TAG -> tag_j at word 0. These are the initial log_alpha values.
    initial_scores_at_pos_0 = torch.zeros(actual_num_tags, device=device)

    # Unigram contributions at pos 0
    for rule_details in U_RULES:
        observed_elements = [get_word_at_offset_fwd(off, 0, sentence_words, num_words) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        for j_idx, current_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"{rule_details['prefix']}:{observed_value_str}_{current_tag_str}"
            if feature_key in feature_to_idx:
                initial_scores_at_pos_0[j_idx] += weights[feature_to_idx[feature_key]]

    # Bigram contributions (START_TAG -> current_tag_str) at pos 0
    for rule_details in B_OBS_RULES:
        observed_elements = [get_word_at_offset_fwd(off, 0, sentence_words, num_words) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        for j_idx, current_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"{rule_details['prefix']}:{observed_value_str}_{START_TAG}_{current_tag_str}"
            if feature_key in feature_to_idx:
                initial_scores_at_pos_0[j_idx] += weights[feature_to_idx[feature_key]]

    if HAS_PURE_B_RULE:
        for j_idx, current_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"B:{START_TAG}_{current_tag_str}"
            if feature_key in feature_to_idx:
                initial_scores_at_pos_0[j_idx] += weights[feature_to_idx[feature_key]]

    log_alpha_prev[:] = initial_scores_at_pos_0

    # --- Recursion step (t=1 to num_words-1) ---
    for i in range(1, num_words):
        log_alpha_current = torch.full((actual_num_tags,), -float('inf'), device=device)

        transition_scores_at_i = torch.zeros((actual_num_tags, actual_num_tags), device=device)
        unigram_scores_at_i = torch.zeros(actual_num_tags, device=device)

        # Unigram contributions for current_tag_str at position i
        for rule_details in U_RULES:
            observed_elements = [get_word_at_offset_fwd(off, i, sentence_words, num_words) for off in rule_details['offsets']]
            observed_value_str = "/".join(observed_elements)
            for j_idx, current_tag_str in enumerate(actual_tag_list_py):
                feature_key = f"{rule_details['prefix']}:{observed_value_str}_{current_tag_str}"
                if feature_key in feature_to_idx:
                    unigram_scores_at_i[j_idx] += weights[feature_to_idx[feature_key]]

        # Bigram contributions (prev_tag_str -> current_tag_str) at position i
        for rule_details in B_OBS_RULES:
            observed_elements = [get_word_at_offset_fwd(off, i, sentence_words, num_words) for off in rule_details['offsets']]
            observed_value_str = "/".join(observed_elements)
            for k_idx, prev_tag_str in enumerate(actual_tag_list_py):
                for j_idx, current_tag_str in enumerate(actual_tag_list_py):
                    feature_key = f"{rule_details['prefix']}:{observed_value_str}_{prev_tag_str}_{current_tag_str}"
                    if feature_key in feature_to_idx:
                        transition_scores_at_i[k_idx, j_idx] += weights[feature_to_idx[feature_key]]

        if HAS_PURE_B_RULE:
            for k_idx, prev_tag_str in enumerate(actual_tag_list_py):
                for j_idx, current_tag_str in enumerate(actual_tag_list_py):
                    feature_key = f"B:{prev_tag_str}_{current_tag_str}"
                    if feature_key in feature_to_idx:
                        transition_scores_at_i[k_idx, j_idx] += weights[feature_to_idx[feature_key]]

        # current_word_potentials[k,j] = B_pure(k,j) + B_obs(obs_i, k,j) + U(obs_i, j)
        current_word_potentials = transition_scores_at_i + unigram_scores_at_i.unsqueeze(0)

        # log_alpha_current[j] = logsumexp_k (log_alpha_prev[k] + current_word_potentials[k, j])
        log_alpha_prev_expanded = log_alpha_prev.unsqueeze(1) # (K) -> (K, 1)
        scores_for_logsumexp = log_alpha_prev_expanded + current_word_potentials # Broadcasts to (K, J)

        log_alpha_current = torch.logsumexp(scores_for_logsumexp, dim=0) # logsumexp over k (dim=0) for each j
        log_alpha_prev = log_alpha_current

    # --- Termination step: Transition to STOP_TAG ---
    scores_to_stop_tag = torch.zeros(actual_num_tags, device=device)

    # Bigram contributions to STOP_TAG
    for rule_details in B_OBS_RULES:
        observed_elements = [get_word_at_offset_fwd(off, num_words, sentence_words, num_words) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        for k_idx, prev_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"{rule_details['prefix']}:{observed_value_str}_{prev_tag_str}_{STOP_TAG}"
            if feature_key in feature_to_idx:
                scores_to_stop_tag[k_idx] += weights[feature_to_idx[feature_key]]

    if HAS_PURE_B_RULE:
        for k_idx, prev_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"B:{prev_tag_str}_{STOP_TAG}"
            if feature_key in feature_to_idx:
                scores_to_stop_tag[k_idx] += weights[feature_to_idx[feature_key]]

    # Unigram contributions for STOP_TAG itself (at position num_words)
    unigram_score_for_stop_itself = torch.tensor(0.0, device=device)
    for rule_details in U_RULES:
        observed_elements = [get_word_at_offset_fwd(off, num_words, sentence_words, num_words) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        feature_key = f"{rule_details['prefix']}:{observed_value_str}_{STOP_TAG}"
        if feature_key in feature_to_idx:
            unigram_score_for_stop_itself += weights[feature_to_idx[feature_key]]

    scores_to_stop_tag += unigram_score_for_stop_itself # Add to all transitions to STOP

    final_terms_for_logsumexp = log_alpha_prev + scores_to_stop_tag
    if not final_terms_for_logsumexp.numel() > 0: # Check if the tensor is empty
        return torch.tensor(-float('inf'), device=device)

    log_Z_x = torch.logsumexp(final_terms_for_logsumexp, dim=0)
    return log_Z_x


def viterbi_decode(sentence_words, weights,
                   feature_to_idx, tag_to_idx, idx_to_tag, device):
    """Finds the best tag sequence using the Viterbi algorithm."""
    num_words = len(sentence_words)
    actual_tag_list_py = [tag for tag in idx_to_tag.values() if tag not in [START_TAG, STOP_TAG]]
    actual_num_tags = len(actual_tag_list_py)

    if actual_num_tags == 0 : return [], torch.tensor(-float('inf'), device=device)

    dp_table = torch.full((num_words, actual_num_tags), -float('inf'), device=device)
    backpointer_table = torch.zeros((num_words, actual_num_tags), dtype=torch.long, device=device)

    if num_words == 0:
        # Handle empty sentence case: Score of START_TAG -> STOP_TAG
        active_indices_empty_stop = get_feature_vector([], 0, STOP_TAG, START_TAG,
                                                  feature_to_idx, tag_to_idx, training_pass=False)
        best_score = calculate_score_for_features(active_indices_empty_stop, weights, device)
        return [], best_score

    def get_word_at_offset_viterbi(offset, current_pos, words, length):
        abs_idx = current_pos + offset
        if 0 <= abs_idx < length: return words[abs_idx]
        elif abs_idx < 0: return "BOS"
        else: return "EOS"

    # --- Initialization step (t=0) ---
    # Scores for START_TAG -> tag_j at word 0
    emissions_at_pos_0 = torch.zeros(actual_num_tags, device=device)

    # Unigram contributions at position 0
    for rule_details in U_RULES:
        observed_elements = [get_word_at_offset_viterbi(off, 0, sentence_words, num_words) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        for j_idx, current_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"{rule_details['prefix']}:{observed_value_str}_{current_tag_str}"
            if feature_key in feature_to_idx:
                emissions_at_pos_0[j_idx] += weights[feature_to_idx[feature_key]]

    # Bigram contributions (START_TAG -> current_tag_str)
    for rule_details in B_OBS_RULES:
        observed_elements = [get_word_at_offset_viterbi(off, 0, sentence_words, num_words) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        for j_idx, current_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"{rule_details['prefix']}:{observed_value_str}_{START_TAG}_{current_tag_str}"
            if feature_key in feature_to_idx:
                emissions_at_pos_0[j_idx] += weights[feature_to_idx[feature_key]]

    if HAS_PURE_B_RULE:
        for j_idx, current_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"B:{START_TAG}_{current_tag_str}"
            if feature_key in feature_to_idx:
                emissions_at_pos_0[j_idx] += weights[feature_to_idx[feature_key]]

    dp_table[0, :] = emissions_at_pos_0

    # --- Recursion step (t=1 to num_words-1) ---
    for i in range(1, num_words):
        transition_scores_at_i = torch.zeros((actual_num_tags, actual_num_tags), device=device)
        unigram_scores_at_i = torch.zeros(actual_num_tags, device=device)

        # Unigram contributions (depend only on current_tag_str and observation at i)
        for rule_details in U_RULES:
            observed_elements = [get_word_at_offset_viterbi(off, i, sentence_words, num_words) for off in rule_details['offsets']]
            observed_value_str = "/".join(observed_elements)
            for j_idx, current_tag_str in enumerate(actual_tag_list_py):
                feature_key = f"{rule_details['prefix']}:{observed_value_str}_{current_tag_str}"
                if feature_key in feature_to_idx:
                    unigram_scores_at_i[j_idx] += weights[feature_to_idx[feature_key]]

        # Bigram contributions (depend on prev_tag_str, current_tag_str, and observation at i)
        for rule_details in B_OBS_RULES:
            observed_elements = [get_word_at_offset_viterbi(off, i, sentence_words, num_words) for off in rule_details['offsets']]
            observed_value_str = "/".join(observed_elements)
            for k_idx, prev_tag_str in enumerate(actual_tag_list_py):
                for j_idx, current_tag_str in enumerate(actual_tag_list_py):
                    feature_key = f"{rule_details['prefix']}:{observed_value_str}_{prev_tag_str}_{current_tag_str}"
                    if feature_key in feature_to_idx:
                        transition_scores_at_i[k_idx, j_idx] += weights[feature_to_idx[feature_key]]

        if HAS_PURE_B_RULE:
            for k_idx, prev_tag_str in enumerate(actual_tag_list_py):
                for j_idx, current_tag_str in enumerate(actual_tag_list_py):
                    feature_key = f"B:{prev_tag_str}_{current_tag_str}"
                    if feature_key in feature_to_idx:
                        transition_scores_at_i[k_idx, j_idx] += weights[feature_to_idx[feature_key]]

        # current_word_potentials[k,j] = B_pure(k,j) + B_obs(obs_i, k,j) + U(obs_i, j)
        current_word_potentials = transition_scores_at_i + unigram_scores_at_i.unsqueeze(0)

        # Viterbi update: dp_table[i, j] = max_k (dp_table[i-1, k] + current_word_potentials[k, j])
        prev_dp_scores_expanded = dp_table[i-1, :].unsqueeze(1) # (K) -> (K, 1)
        combined_scores = prev_dp_scores_expanded + current_word_potentials # (K, 1) + (K, J) -> (K, J)

        max_scores, best_prev_indices = torch.max(combined_scores, dim=0) # Max over k (rows) for each j (column)

        dp_table[i, :] = max_scores
        backpointer_table[i, :] = best_prev_indices

    # --- Termination step: Transition to STOP_TAG ---
    scores_to_stop_tag = torch.zeros(actual_num_tags, device=device)

    # Bigram contributions (prev_tag_str -> STOP_TAG)
    for rule_details in B_OBS_RULES:
        observed_elements = [get_word_at_offset_viterbi(off, num_words, sentence_words, num_words) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        for k_idx, prev_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"{rule_details['prefix']}:{observed_value_str}_{prev_tag_str}_{STOP_TAG}"
            if feature_key in feature_to_idx:
                scores_to_stop_tag[k_idx] += weights[feature_to_idx[feature_key]]

    if HAS_PURE_B_RULE:
        for k_idx, prev_tag_str in enumerate(actual_tag_list_py):
            feature_key = f"B:{prev_tag_str}_{STOP_TAG}"
            if feature_key in feature_to_idx:
                scores_to_stop_tag[k_idx] += weights[feature_to_idx[feature_key]]

    # Unigram contributions for STOP_TAG itself
    unigram_score_for_stop_itself = torch.tensor(0.0, device=device)
    for rule_details in U_RULES:
        observed_elements = [get_word_at_offset_viterbi(off, num_words, sentence_words, num_words) for off in rule_details['offsets']]
        observed_value_str = "/".join(observed_elements)
        feature_key = f"{rule_details['prefix']}:{observed_value_str}_{STOP_TAG}"
        if feature_key in feature_to_idx:
            unigram_score_for_stop_itself += weights[feature_to_idx[feature_key]]

    scores_to_stop_tag += unigram_score_for_stop_itself # Add to all transitions to STOP

    final_scores_before_stop = dp_table[num_words-1, :] + scores_to_stop_tag
    max_final_score, best_last_tag_actual_idx = torch.max(final_scores_before_stop, dim=0)

    # --- Path reconstruction (backtracking) ---
    best_path_indices_gpu = torch.empty(num_words, dtype=torch.long, device=device)
    best_path_indices_gpu[num_words-1] = best_last_tag_actual_idx

    current_best_tag_idx_for_bp = best_last_tag_actual_idx
    for i in range(num_words - 1, 0, -1):
        prev_tag_actual_idx = backpointer_table[i, current_best_tag_idx_for_bp]
        best_path_indices_gpu[i-1] = prev_tag_actual_idx
        current_best_tag_idx_for_bp = prev_tag_actual_idx

    predicted_tags_str = [actual_tag_list_py[idx.item()] for idx in best_path_indices_gpu]
    return predicted_tags_str, max_final_score


def train(input_file, model_output_path, num_epochs=10, learning_rate=0.01):
    print(f"Starting CRF training...")
    print(f"Input: {input_file}, Model Output: {model_output_path}")
    print(f"Epochs: {num_epochs}, LR: {learning_rate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    sentences_words = []
    sentences_tags_str = []
    tag_set = set([START_TAG, STOP_TAG])

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
            word, tag = parts[0], parts[-1] # Assumes tag is the last part
            current_words.append(word)
            current_tags.append(tag)
            tag_set.add(tag)
        if current_words: # Add last sentence
            sentences_words.append(current_words)
            sentences_tags_str.append(current_tags)

    tag_to_idx = {tag: i for i, tag in enumerate(list(tag_set))}
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}

    feature_to_idx = {}
    print("Building feature map from training data (true paths)...")
    for i in range(len(sentences_words)):
        sentence_w = sentences_words[i]
        sentence_t_str = sentences_tags_str[i]
        for j in range(len(sentence_w)):
            curr_t = sentence_t_str[j]
            prev_t = sentence_t_str[j-1] if j > 0 else START_TAG
            get_feature_vector(sentence_w, j, curr_t, prev_t, feature_to_idx, tag_to_idx, training_pass=True)
        # For STOP_TAG transition
        if sentence_w:
            get_feature_vector(sentence_w, len(sentence_w), STOP_TAG, sentence_t_str[-1], feature_to_idx, tag_to_idx, training_pass=True)
        else: # Empty sentence
             get_feature_vector([], 0, STOP_TAG, START_TAG, feature_to_idx, tag_to_idx, training_pass=True)

    # Add all pure "B:prev_curr" transitions to feature_to_idx if "B" template rule exists.
    if HAS_PURE_B_RULE:
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
        model_data = {
            'weights': torch.tensor([]),
            'feature_to_idx': {}, 'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
        }
        with open(model_output_path, 'wb') as f_model:
            pickle.dump(model_data, f_model)
        print(f"Empty model saved to {model_output_path}")
        return

    weights = nn.Parameter(torch.zeros(num_features, dtype=torch.float32, device=device))
    optimizer = optim.SGD([weights], lr=learning_rate)

    print("Starting training loop...")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_epoch_loss = 0.0
        sentence_iterator = tqdm(range(len(sentences_words)), desc=f"Epoch {epoch+1}/{num_epochs} Sentences", leave=False)
        for i in sentence_iterator:
            optimizer.zero_grad()

            words = sentences_words[i]
            true_tags = sentences_tags_str[i]

            s_gold = sentence_score(words, true_tags, weights, feature_to_idx, tag_to_idx, device)
            log_Z = log_forward_algorithm(words, weights, feature_to_idx, tag_to_idx, idx_to_tag, device)

            # Loss = log Z(x) - Score(y_true, x)
            loss = log_Z - s_gold

            if not torch.isinf(loss) and not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                current_loss_item = loss.item()
                total_epoch_loss += current_loss_item
                wandb.log({"sentence_loss": current_loss_item, "sentence_idx": i, "epoch": epoch + 1})
            else:
                current_loss_item = loss.item()
                print(f"Warning: Skipping backward pass for sentence {i} due to inf/nan loss: {current_loss_item}")
                wandb.log({"sentence_loss_skipped": current_loss_item, "sentence_idx": i, "epoch": epoch + 1})

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Sentence {i+1}/{len(sentences_words)}, Current Loss: {current_loss_item:.4f}")

        avg_epoch_loss = total_epoch_loss / len(sentences_words) if len(sentences_words) > 0 else 0
        tqdm.write(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_average_loss": avg_epoch_loss, "epoch": epoch + 1})

    # Save model
    model_data = {
        'weights': weights.data.cpu(), # Save weights on CPU for portability
        'feature_to_idx': feature_to_idx,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag,
    }
    with open(model_output_path, 'wb') as f_model:
        pickle.dump(model_data, f_model)
    print(f"Training complete. Model saved to {model_output_path}")
    wandb.finish()


def predict(model_input_path, input_file, output_file):
    print(f"Starting CRF prediction...")
    print(f"Model: {model_input_path}, Input: {input_file}, Output: {output_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for prediction: {device}")

    with open(model_input_path, 'rb') as f_model:
        model_data = pickle.load(f_model)

    weights = model_data['weights'].to(device) # Move to current device
    feature_to_idx = model_data['feature_to_idx']
    tag_to_idx = model_data['tag_to_idx']
    idx_to_tag = model_data['idx_to_tag']

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
            # Input for prediction can be just words or word + other info; word is the first part.
            parts = line.split()
            if parts:
                 current_words.append(parts[0])
        if current_words:
            sentences_words.append(current_words)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for words in tqdm(sentences_words, desc="Predicting Sentences"):
            if not words:
                f_out.write("\n")
                continue

            predicted_tags, _ = viterbi_decode(words, weights, feature_to_idx, tag_to_idx, idx_to_tag, device)

            for word, tag in zip(words, predicted_tags):
                f_out.write(f"{word} {tag}\n")
            f_out.write("\n")

    print(f"Prediction complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear-chain CRF for NER")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the CRF model")
    train_parser.add_argument("--input", required=True, help="Path to the training data file (CoNLL format)")
    train_parser.add_argument("--model", required=True, help="Path to save the trained model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

    predict_parser = subparsers.add_parser("predict", help="Predict tags using a trained CRF model")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model file")
    predict_parser.add_argument("--input", required=True, help="Path to the input file for prediction (words per line)")
    predict_parser.add_argument("--output", required=True, help="Path to save the prediction results")

    args = parser.parse_args()

    if args.command == "train":
        train(args.input, args.model, num_epochs=args.epochs, learning_rate=args.lr)
    elif args.command == "predict":
        predict(args.model, args.input, args.output) 
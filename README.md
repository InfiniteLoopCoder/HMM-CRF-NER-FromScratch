# Named Entity Recognition (NER) Models

This repository contains implementations of three different models for Named Entity Recognition (NER):
1.  **Hidden Markov Model (HMM)**: A classical statistical model.
2.  **Linear Chain Conditional Random Field (CRF)**: A discriminative model that builds upon features from the input sequence.
3.  **Transformer-CRF**: A neural network model combining a Transformer encoder with a CRF layer for state-of-the-art performance.

The models are trained and evaluated on CoNLL-formatted data.

## Project Structure

```
.
├── English/              # Directory containing example English NER data
│   ├── train.txt         # Example training data
│   ├── validation.txt    # Example validation data
│   └── tag.txt           # List of possible tags
├── crf_ner.py            # Implementation of the Linear Chain CRF model
├── hmm_ner.py            # Implementation of the Hidden Markov Model
├── transformer_crf_ner.py # Implementation of the Transformer-CRF model
└── README.md             # This file
```

## Setup

### Prerequisites

*   Python 3.10
*   PyTorch (refer to [pytorch.org](https://pytorch.org/) for installation instructions based on your system and CUDA version if applicable)
*   Weights & Biases (wandb) for experiment tracking (optional, used by CRF and Transformer-CRF): `pip install wandb`
*   tqdm for progress bars: `pip install tqdm`

### Data Format

The input data for training and prediction should be in a CoNLL-like format. Each line contains a word and its corresponding tag, separated by a space. Sentences are separated by an empty line.

**Example `train.txt` / `validation.txt`:**

```
EU O
rejects O
German B-MISC
call O
to O
boycott O
British B-MISC
lamb O
. O

Peter B-PER
Blackburn I-PER
```

**Prediction Input File Format:**
For prediction, the input file should contain one word per line. Tags, if present, will be ignored. Sentences are separated by an empty line.

**Example `predict_input.txt`:**

```
British
lamb
and
Peter
Blackburn
```

**Output Format (for prediction):**
The prediction output will be in the same CoNLL-like format as the training data, with words and their predicted tags.

```
British B-MISC
lamb O
and O
Peter B-PER
Blackburn I-PER
```

## Model Implementations

### 1. Hidden Markov Model (`hmm_ner.py`)

A generative statistical model that learns transition probabilities between tags and emission probabilities of words given tags.

**Training:**

```bash
python hmm_ner.py train --input ./English/train.txt --model hmm_model.pkl
```

*   `--input`: Path to the training data file.
*   `--model`: Path to save the trained HMM model (pickle file).

**Prediction:**

```bash
python hmm_ner.py predict --model hmm_model.pkl --input ./English/validation.txt --output hmm_predictions.txt
```

*   `--model`: Path to the trained HMM model file.
*   `--input`: Path to the input file for prediction (words per line).
*   `--output`: Path to save the prediction results.

### 2. Linear Chain Conditional Random Field (`crf_ner.py`)

A discriminative model that directly models the conditional probability of a tag sequence given an observation sequence. It uses hand-crafted feature templates.

**Features:**
The CRF implementation uses a set of predefined unigram and bigram feature templates based on words and their surrounding context (offsets `[-2, -1, 0, 1, 2]`). Pure tag-to-tag transition features are also included.

**Experiment Tracking:**
This script integrates with Weights & Biases (`wandb`) for logging training progress (loss, epochs, etc.). Ensure you have `wandb` installed and configured (`wandb login`).

**Training:**

```bash
python crf_ner.py train --input ./English/train.txt --model crf_model.pkl --epochs 20 --lr 0.01
```

*   `--input`: Path to the training data file.
*   `--model`: Path to save the trained CRF model (pickle file).
*   `--epochs` (optional): Number of training epochs (default: 10).
*   `--lr` (optional): Learning rate (default: 0.01).

**Prediction:**

```bash
python crf_ner.py predict --model crf_model.pkl --input ./English/validation.txt --output crf_predictions.txt
```

*   `--model`: Path to the trained CRF model file.
*   `--input`: Path to the input file for prediction.
*   `--output`: Path to save the prediction results.

### 3. Transformer-CRF (`transformer_crf_ner.py`)

This model uses a Transformer encoder to learn rich contextual representations of words, followed by a CRF layer to model tag dependencies.

**Key Components:**
*   **Word Embeddings:** Standard learnable embeddings for words.
*   **Positional Encoding:** Adds information about the position of words in the sequence.
*   **Transformer Encoder:** Multiple layers of multi-head self-attention and feed-forward networks to capture context.
*   **CRF Layer:** Learns transition scores between tags and uses Viterbi decoding for prediction.

**Experiment Tracking:**
This script also integrates with Weights & Biases (`wandb`) for experiment tracking.

**Training:**

```bash
python transformer_crf_ner.py train --input ./English/train.txt --model transformer_crf_model.pth --epochs 5 --lr 0.0001
```

*   `--input`: Path to the training data file.
*   `--model`: Path to save the trained Transformer-CRF model (`.pth` file).
*   `--epochs` (optional): Number of training epochs (default: 5).
*   `--lr` (optional): Learning rate for the AdamW optimizer (default: 1e-4).

The `MAX_SEQ_LEN` is dynamically calculated based on the training data (rounded up to the next power of 2 for efficiency). Hyperparameters like embedding dimension, Transformer hidden dimension, number of heads, and layers are defined within the script but can be adjusted.

**Prediction:**

```bash
python transformer_crf_ner.py predict --model transformer_crf_model.pth --input ./English/validation.txt --output transformer_crf_predictions.txt
```

*   `--model`: Path to the trained Transformer-CRF model file.
*   `--input`: Path to the input file for prediction.
*   `--output`: Path to save the prediction results.

## Notes

*   The example data in the `English/` directory is small and intended for quick testing. For robust models, larger datasets are required.
*   The Transformer-CRF model is generally expected to provide the best performance among the three, especially with sufficient training data, due to its ability to learn complex contextual features.
*   Ensure that the tag set implied by your training data is consistent. The `tag.txt` file in `English/` is an example and is not directly used by the scripts (tag sets are derived from the training data).
*   For GPU usage with PyTorch models (CRF and Transformer-CRF), ensure you have a compatible CUDA version and PyTorch build. The scripts will automatically try to use a GPU if available. 
# NLP-Based Authorship Attribution

A Python-based Natural Language Processing (NLP) classifier that identifies likely authorship by analyzing linguistic patterns and writing styles. The program builds statistical text models using five stylistic features and compares unknown writing samples against reference texts through log-based similarity scoring.

## Features

- **Linguistic Feature Extraction** - Analyzes word frequencies, word lengths, word stems, sentence lengths, and pronoun usage.
- **Text Preprocessing** - Cleans and tokenizes text to prepare it for analysis.
- **Custom Word Stemming** - Applies suffix-based rules to identify common word stems.
- **Statistical Text Modeling** - Builds frequency dictionaries to represent linguistic patterns in writing samples.
- **Log-Based Similarity Scoring** - Measures how closely an unknown text matches each reference model.
- **Authorship Classification** - Predicts the more likely source by comparing similarity scores across five linguistic features.
- **Model Persistence** - Supports saving and loading text models for future comparisons.

## How It Works

The classifier analyzes an unknown writing sample and compares it against two reference text models.

1. **Preprocess the text.** Remove common punctuation, convert words to lowercase, and tokenize the input.
2. **Extract linguistic features.** Build frequency distributions for words, word lengths, stems, sentence lengths, and pronouns.
3. **Build reference models.** Analyze known writing samples to create distinct stylistic profiles.
4. **Calculate similarity scores.** Use logarithmic probability-based scoring to compare the unknown text against each reference model.
5. **Compare feature scores.** Each of the five linguistic features contributes one vote to the reference model with the higher score.
6. **Predict authorship.** Select the source with the most votes, or report an inconclusive result if the scores are tied.

### Linguistic Features

| Feature | Description |
| --- | --- |
| Word Frequencies | Tracks how often individual words appear |
| Word Lengths | Measures the distribution of word lengths |
| Word Stems | Groups words using custom suffix-removal rules |
| Sentence Lengths | Tracks the number of words per sentence |
| Pronoun Usage | Measures the frequency of selected pronouns |

Together, these features create a stylistic profile that can be compared with other texts.

## Getting Started

### Prerequisites

- Python 3
- No external libraries required

### Running the Program

**1. Clone the repository**

```bash
git clone https://github.com/gabrielahbadari/NLP-Based-Authorship-Attribution.git
cd NLP-Based-Authorship-Attribution
```

**2. Run the classifier**

```bash
python3 authorship_classifier.py
```

The program runs a built-in demonstration comparing an unknown text against two reference writing samples.

### Using Your Own Texts

Import the `TextModel` class and create models for two reference texts and an unknown writing sample.

```python
from authorship_classifier import TextModel

source1 = TextModel("Author A")
source1.add_file("author_a.txt")

source2 = TextModel("Author B")
source2.add_file("author_b.txt")

unknown = TextModel("Unknown Text")
unknown.add_file("unknown.txt")

unknown.classify(source1, source2)
```

The classifier prints the similarity scores for each reference model and identifies the more likely source.

## Technologies

- Python
- Natural Language Processing (NLP)
- Text Preprocessing and Tokenization
- Statistical Text Modeling
- Log-Based Similarity Scoring
- Dictionary-Based Feature Extraction
"""
authorship_classifier.py

An NLP-based authorship attribution system that identifies writing styles
using linguistic feature extraction and log-based similarity scoring.

The classifier analyzes five stylistic features:
    1. Word frequencies
    2. Word length distributions
    3. Word stems
    4. Sentence length distributions
    5. Pronoun usage

An unknown text is compared against two reference text models to
determine which source has the more similar writing style.
"""

import ast
import math


# Pronouns tracked as a stylistic feature
PRONOUNS = {
    "i", "she", "he", "they", "their", "her", "him", "them",
    "it", "its", "me", "we", "us", "our", "your", "yours",
    "mine", "his", "hers", "herself", "himself", "themselves",
    "itself", "myself", "yourself", "yourselves"
}


# ------------------------------------------------------------
# Text Preprocessing
# ------------------------------------------------------------

def clean_text(text):
    """
    Removes common punctuation, converts text to lowercase,
    and returns a list of words.
    """
    for symbol in """.,?"'!;:""":
        text = text.replace(symbol, "")

    return text.lower().split()


def stem(word):
    """
    Reduces a word to a simplified stem using custom
    suffix-removal rules.

    Handles common endings such as -s, -ing, -er, -y,
    -es, -ed, and -e.
    """
    # Remove plural -s
    if len(word) > 1 and word.endswith("s") and word[-2] != "s":
        word = word[:-1]

        if len(word) > 3 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 1 and word.endswith("e"):
            word = word[:-1]
        elif len(word) > 2 and word.endswith(("ed", "er")):
            word = word[:-2]

    # Remove -ing, accounting for repeated consonants
    if len(word) > 3 and word.endswith("ing"):
        if len(word) > 4 and word[-4] == word[-5]:
            word = word[:-4]
        else:
            word = word[:-3]

    # Remove -er
    elif len(word) > 2 and word.endswith("er"):
        if len(word) > 3 and word[-3] == word[-4]:
            word = word[:-3]
        else:
            word = word[:-2]

    # Replace -y with -i
    elif len(word) > 1 and word.endswith("y"):
        word = word[:-1] + "i"

    # Remove remaining suffixes
    elif len(word) > 2 and word.endswith("es"):
        word = word[:-2]

    elif len(word) > 2 and word.endswith("ed"):
        word = word[:-2]

    elif len(word) > 1 and word.endswith("e"):
        if not word.startswith("th"):
            word = word[:-1]

    return word


# ------------------------------------------------------------
# Similarity Scoring
# ------------------------------------------------------------

def compare_dictionaries(reference, unknown):
    """
    Computes a log-based similarity score between two
    feature-frequency dictionaries.

    Frequencies from the reference model are used to
    estimate the likelihood of features in the unknown text.
    Unseen features receive a small smoothing value.

    A higher score indicates greater similarity.
    """
    if not reference:
        return -50

    total = sum(reference.values())
    score = 0

    for feature, count in unknown.items():
        if feature in reference:
            probability = reference[feature] / total
        else:
            probability = 0.5 / total

        score += math.log(probability) * count

    return score


# ------------------------------------------------------------
# Text Model
# ------------------------------------------------------------

class TextModel:
    """
    Represents the linguistic profile of a text.

    Stores frequency distributions for five stylistic features
    used to compare and classify writing samples.
    """

    def __init__(self, model_name):
        """Initializes an empty text model."""
        self.name = model_name
        self.words = {}
        self.word_lengths = {}
        self.stems = {}
        self.sentence_lengths = {}
        self.pronouns = {}

    def __repr__(self):
        """Returns a summary of the model's feature dictionaries."""
        return (
            f"Text model: {self.name}\n"
            f"  Unique words: {len(self.words)}\n"
            f"  Word lengths: {len(self.word_lengths)}\n"
            f"  Word stems: {len(self.stems)}\n"
            f"  Sentence lengths: {len(self.sentence_lengths)}\n"
            f"  Pronouns: {len(self.pronouns)}"
        )

    def add_string(self, text):
        """
        Analyzes a string and updates all five linguistic
        feature distributions.
        """

        # 1. Sentence length distribution
        sentences = [text]

        for punctuation in [".", "!", "?"]:
            parts = []

            for sentence in sentences:
                parts.extend(sentence.split(punctuation))

            sentences = parts

        for sentence in sentences:
            length = len(sentence.split())

            if length > 0:
                self.sentence_lengths[length] = (
                    self.sentence_lengths.get(length, 0) + 1
                )

        # Clean and tokenize the text
        words = clean_text(text)

        # 2. Word frequency and word length distributions
        for word in words:
            self.words[word] = self.words.get(word, 0) + 1

            length = len(word)
            self.word_lengths[length] = (
                self.word_lengths.get(length, 0) + 1
            )

        # 3. Stem frequency distribution
        for word in words:
            word_stem = stem(word)
            self.stems[word_stem] = (
                self.stems.get(word_stem, 0) + 1
            )

        # 4. Pronoun frequency distribution
        for word in words:
            if word in PRONOUNS:
                self.pronouns[word] = (
                    self.pronouns.get(word, 0) + 1
                )

    def add_file(self, filename):
        """
        Reads a text file and incorporates its contents
        into the model's linguistic distributions.
        """
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

        self.add_string(text)

    # --------------------------------------------------------
    # Model Persistence
    # --------------------------------------------------------

    def save_model(self):
        """
        Saves each linguistic feature dictionary to a
        separate file for later use.
        """
        features = {
            "words": self.words,
            "word_lengths": self.word_lengths,
            "stems": self.stems,
            "sentence_lengths": self.sentence_lengths,
            "pronouns": self.pronouns
        }

        for feature_name, dictionary in features.items():
            filename = f"{self.name}_{feature_name}"

            with open(filename, "w", encoding="utf-8") as file:
                file.write(str(dictionary))

    def read_model(self):
        """
        Loads previously saved linguistic feature dictionaries.

        Uses ast.literal_eval() to safely parse stored
        dictionary representations.
        """
        feature_names = [
            "words",
            "word_lengths",
            "stems",
            "sentence_lengths",
            "pronouns"
        ]

        for feature_name in feature_names:
            filename = f"{self.name}_{feature_name}"

            with open(filename, "r", encoding="utf-8") as file:
                dictionary = ast.literal_eval(file.read())

            setattr(self, feature_name, dictionary)

    # --------------------------------------------------------
    # Authorship Classification
    # --------------------------------------------------------

    def similarity_scores(self, other):
        """
        Computes five log-based similarity scores between
        this model and a reference text model.

        Returns scores for words, word lengths, stems,
        sentence lengths, and pronouns.
        """
        return [
            compare_dictionaries(other.words, self.words),
            compare_dictionaries(other.word_lengths, self.word_lengths),
            compare_dictionaries(other.stems, self.stems),
            compare_dictionaries(
                other.sentence_lengths, self.sentence_lengths
            ),
            compare_dictionaries(other.pronouns, self.pronouns)
        ]

    def classify(self, source1, source2):
        """
        Compares an unknown text model against two reference models.

        Each linguistic feature awards one point to the source
        with the higher similarity score. The source winning
        the most feature comparisons is selected.
        """
        scores1 = self.similarity_scores(source1)
        scores2 = self.similarity_scores(source2)

        print(f"Scores for {source1.name}: {scores1}")
        print(f"Scores for {source2.name}: {scores2}")

        source1_wins = 0
        source2_wins = 0

        for score1, score2 in zip(scores1, scores2):
            if score1 > score2:
                source1_wins += 1
            elif score2 > score1:
                source2_wins += 1

        if source1_wins > source2_wins:
            predicted_source = source1.name

        elif source2_wins > source1_wins:
            predicted_source = source2.name

        else:
            print("Classification inconclusive: tied similarity scores.")
            return None

        print(f"{self.name} is more likely to come from {predicted_source}.")
        return predicted_source


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

def main():
    """Demonstrates authorship classification with sample texts."""

    source1 = TextModel("Source 1")
    source1.add_string("It is interesting that she is interested.")

    source2 = TextModel("Source 2")
    source2.add_string("I am very, very excited about this!")

    unknown = TextModel("Unknown Text")
    unknown.add_string("Is he interested? No, but I am.")

    unknown.classify(source1, source2)


if __name__ == "__main__":
    main()

"""Lexicon-based sentiment classifier with negation handling.

This is the *classical NLP baseline*. The LLM results are compared against it
in the evaluation step so the benefit of the LLM can be measured.
"""
from src.preprocess import tokenize

POSITIVE = {
    "good", "great", "excellent", "amazing", "awesome", "love", "loved", "superb",
    "wonderful", "fantastic", "perfect", "best", "happy", "delicious", "friendly",
    "comfortable", "fast", "quick", "recommended", "worth", "nice", "helpful",
    "polite", "beautiful", "gorgeous", "impressive", "reliable", "smooth", "easy",
}
NEGATIVE = {
    "bad", "terrible", "worst", "poor", "awful", "horrible", "hate", "disappointed",
    "disappointing", "late", "damaged", "broke", "broken", "waste", "cheap",
    "annoying", "slow", "useless", "rude", "regret", "faulty", "defective", "ugly",
}
NEGATIONS = {"not", "no", "never", "dont", "didnt", "isnt", "wasnt", "cant", "wont",
             "wouldnt", "doesnt", "hardly", "barely", "cannot"}


def score_tokens(tokens: list) -> int:
    """+1 for every positive word, -1 for every negative word; a negation
    within the previous two tokens flips the polarity."""
    total = 0
    for i, tok in enumerate(tokens):
        polarity = 1 if tok in POSITIVE else -1 if tok in NEGATIVE else 0
        if polarity and any(t in NEGATIONS for t in tokens[max(0, i - 2):i]):
            polarity = -polarity
        total += polarity
    return total


def predict(text: str) -> str:
    """Return 'positive', 'negative' or 'neutral'."""
    score = score_tokens(tokenize(text))
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"

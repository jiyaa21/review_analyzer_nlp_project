"""Classical NLP preprocessing: cleaning, tokenisation, stop-word removal, keywords."""
import re
from collections import Counter

# Small built-in stop-word list (no external download needed -> sandbox friendly).
# Negation words (not, no, never ...) are deliberately NOT included.
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "this", "that", "with", "have", "has",
    "had", "you", "your", "its", "it's", "but", "from", "they", "them", "their", "will",
    "just", "very", "all", "can", "could", "would", "should", "been", "being", "than",
    "then", "there", "here", "when", "what", "which", "who", "whom", "also", "about",
    "after", "before", "into", "out", "over", "under", "again", "does", "did", "doing",
    "one", "two", "get", "got", "any", "some", "our", "ours", "his", "her", "she", "him",
    "ive", "its", "off", "too", "much", "more", "most", "such", "only", "own", "same",
    "while", "because", "within", "even", "ever", "still", "after", "supposed",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z']+")


def clean_text(text: str) -> str:
    """Lower-case, strip URLs / HTML tags and collapse whitespace."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list:
    """Split cleaned text into word tokens; apostrophes are removed (didn't -> didnt)."""
    return [t.replace("'", "") for t in _TOKEN_RE.findall(clean_text(text)) if t.strip("'")]


def remove_stopwords(tokens: list, min_len: int = 3) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) >= min_len]


def extract_keywords(text: str, top_n: int = 5, min_len: int = 3) -> list:
    """Return the most frequent content words of a text."""
    tokens = remove_stopwords(tokenize(text), min_len)
    return [word for word, _ in Counter(tokens).most_common(top_n)]


def corpus_keywords(texts: list, top_n: int = 10, min_len: int = 3) -> list:
    """Most frequent content words across many texts, as (word, count) pairs."""
    counter = Counter()
    for text in texts:
        counter.update(remove_stopwords(tokenize(text), min_len))
    return counter.most_common(top_n)

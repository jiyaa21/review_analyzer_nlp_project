"""Run with:  python -m unittest discover tests   (no API key needed)"""
import unittest

from src.analyzer import parse_llm_json, validate_analysis
from src.baseline import predict
from src.evaluate import evaluate
from src.preprocess import clean_text, extract_keywords, tokenize


class TestPreprocess(unittest.TestCase):
    def test_clean_removes_url_and_html(self):
        self.assertEqual(clean_text("Great <b>phone</b> http://x.com  !"), "great phone !")

    def test_tokenize_apostrophe(self):
        self.assertIn("didnt", tokenize("It didn't work"))

    def test_keywords(self):
        self.assertEqual(extract_keywords("battery battery screen the", 1), ["battery"])


class TestBaseline(unittest.TestCase):
    def test_negation(self):
        self.assertEqual(predict("This is not good"), "negative")
        self.assertEqual(predict("Great product, love it"), "positive")


class TestParsing(unittest.TestCase):
    def test_fenced_json(self):
        data = parse_llm_json('```json\n{"sentiment": "Positive"}\n```')
        self.assertEqual(validate_analysis(data)["sentiment"], "positive")

    def test_invalid_sentiment_rejected(self):
        with self.assertRaises(ValueError):
            validate_analysis({"sentiment": "happy"})


class TestMetrics(unittest.TestCase):
    def test_perfect_score(self):
        y = ["positive", "negative", "neutral"]
        self.assertEqual(evaluate(y, y)["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

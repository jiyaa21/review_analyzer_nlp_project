"""Review Insight Analyzer - command line interface.

Examples
    python main.py single "The battery is great but the screen is dull"
    python main.py analyze --limit 10
    python main.py evaluate
    python main.py analyze --mock          # offline test, no API key needed
"""
import argparse
import csv
import json
import os
import sys

import yaml
from dotenv import load_dotenv

from src.analyzer import ReviewAnalyzer, load_prompts
from src.baseline import predict as baseline_predict
from src.evaluate import evaluate
from src.llm_client import LLMClient, LLMError, MockLLMClient


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_reviews(path: str, limit: int = None) -> list:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "review" not in rows[0]:
        raise ValueError(f"{path} must be a CSV with at least a 'review' column")
    for i, row in enumerate(rows, 1):
        row.setdefault("id", str(i))
    return rows[:limit] if limit else rows


def build_analyzer(cfg: dict, mock: bool) -> ReviewAnalyzer:
    prompts = load_prompts(cfg["paths"]["prompt_file"])
    if mock:
        cfg["llm"]["request_delay"] = 0.0
        print("[mock mode] Using the offline fake LLM - results are NOT from a real model.\n")
        return ReviewAnalyzer(MockLLMClient(), prompts, cfg, use_cache=False)
    return ReviewAnalyzer(LLMClient(cfg["llm"]), prompts, cfg, use_cache=True)


def save_results(results: list, report: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "review", "sentiment", "confidence", "aspects",
                         "keywords", "summary", "source"])
        for r in results:
            writer.writerow([r["id"], r["review"], r["sentiment"], r["confidence"],
                             "; ".join(f"{a['aspect']}:{a['sentiment']}" for a in r["aspects"]),
                             ", ".join(r["keywords"]), r["summary"], r["source"]])
    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return csv_path, json_path


def cmd_single(args, cfg):
    analyzer = build_analyzer(cfg, args.mock)
    res = analyzer.analyze_one(args.text)
    print(json.dumps(res, indent=2))


def cmd_analyze(args, cfg):
    reviews = read_reviews(args.input or cfg["paths"]["input_csv"], args.limit)
    analyzer = build_analyzer(cfg, args.mock)

    def progress(i, total, res):
        print(f"[{i:>2}/{total}] {res['sentiment']:<8} ({res['source']:<8}) {res['review'][:60]}")

    results = analyzer.analyze_batch(reviews, progress)
    stats = analyzer.build_statistics(results)
    report_text = analyzer.generate_report(stats)
    report = {"statistics": stats, "llm_report": report_text,
              "api_calls": analyzer.api_calls, "model": analyzer.client.model}
    csv_path, json_path = save_results(results, report, cfg["paths"]["output_dir"])

    print("\n=== Sentiment distribution ===")
    for label, n in sorted(stats["sentiment_counts"].items()):
        print(f"  {label:<9}: {n}")
    print("\n=== Business report ===\n" + report_text)
    print(f"\nAPI calls made: {analyzer.api_calls} | saved: {csv_path}, {json_path}")


def cmd_evaluate(args, cfg):
    reviews = read_reviews(args.input or cfg["paths"]["input_csv"], args.limit)
    if "label" not in reviews[0]:
        sys.exit("Evaluation needs a 'label' column in the CSV.")
    analyzer = build_analyzer(cfg, args.mock)
    results = analyzer.analyze_batch(reviews)

    truth = [r["label"].strip().lower() for r in reviews]
    llm_pred = [r["sentiment"] for r in results]
    base_pred = [baseline_predict(r["review"]) for r in reviews]
    report = {"model": analyzer.client.model, "samples": len(reviews),
              "lexicon_baseline": evaluate(truth, base_pred),
              "llm": evaluate(truth, llm_pred),
              "fallback_count": sum(r["source"] == "fallback" for r in results)}

    print(f"{'Method':<18}{'Accuracy':>10}{'Macro-F1':>10}")
    print(f"{'Lexicon baseline':<18}{report['lexicon_baseline']['accuracy']:>10}{report['lexicon_baseline']['macro_f1']:>10}")
    print(f"{'LLM (' + analyzer.client.model[:10] + ')':<18}{report['llm']['accuracy']:>10}{report['llm']['macro_f1']:>10}")
    print(f"\nLLM confusion matrix (rows = truth): {json.dumps(report['llm']['confusion_matrix'])}")
    if report["fallback_count"]:
        print(f"Warning: {report['fallback_count']} reviews used the fallback (LLM failed).")

    out = os.path.join(cfg["paths"]["output_dir"], "evaluation.json")
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Review Insight Analyzer (NLP + LLM)")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", help="analyse one review text")
    p_single.add_argument("text")
    p_single.add_argument("--mock", action="store_true", help="offline fake LLM")
    p_single.set_defaults(func=cmd_single)

    for name, func, helptext in (("analyze", cmd_analyze, "analyse a CSV of reviews"),
                                 ("evaluate", cmd_evaluate, "compare LLM vs baseline on labelled data")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--input", help="CSV path (default from config.yaml)")
        p.add_argument("--limit", type=int, help="only process the first N reviews")
        p.add_argument("--mock", action="store_true", help="offline fake LLM")
        p.set_defaults(func=func)

    args = parser.parse_args()
    load_dotenv()
    try:
        args.func(args, load_config(args.config))
    except (LLMError, FileNotFoundError, ValueError, KeyError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()

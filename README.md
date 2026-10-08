# Review Insight Analyzer (NLP + LLM)

A beginner-friendly NLP project that mines customer reviews using **classical NLP
preprocessing** together with an **LLM (Llama 3.1 via the Groq API)**.

For every review it produces: overall **sentiment**, **confidence**, **aspect-level
sentiment** (e.g. battery: negative, camera: positive), a one-line **summary** and
**keywords**. For the whole batch it generates a short **business report**.
It also **evaluates** the LLM against a lexicon-based baseline on labelled data.

## Why an LLM here?
A lexicon baseline fails on sarcasm ("Great, it broke in a day"), mixed reviews and
aspect extraction. The LLM handles these, and the project measures the difference.

## Pipeline
```
reviews.csv -> clean/tokenise/keywords (preprocess.py)
            -> prompt from prompts/review_prompts.yaml
            -> Groq LLM API (llm_client.py, retries + back-off)
            -> JSON parsing + schema validation (analyzer.py)
            -> cache / fallback to baseline on failure
            -> statistics -> LLM report -> outputs/ (CSV + JSON)
```

## Project structure
| Path | Purpose |
|------|---------|
| `main.py` | CLI (`single`, `analyze`, `evaluate`) |
| `config.yaml` | model, API URL, temperature, retries, file paths |
| `prompts/review_prompts.yaml` | system, analysis (few-shot) and report prompts |
| `src/preprocess.py` | cleaning, tokenisation, stop-words, keywords |
| `src/baseline.py` | lexicon + negation sentiment baseline |
| `src/llm_client.py` | Groq API client (+ offline `MockLLMClient`) |
| `src/analyzer.py` | prompt building, validation, cache, aggregation |
| `src/evaluate.py` | accuracy / precision / recall / F1 / confusion matrix |
| `data/reviews.csv` | 24 labelled sample reviews |
| `tests/test_basic.py` | unit tests (no API key needed) |

## Setup
1. Get a free API key: https://console.groq.com/keys
2. Install and configure:
```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your key in .env
```
(In CodeSandbox: open the terminal, run the same commands. Alternatively add
`GROQ_API_KEY` under *Server Control Panel -> Secrets / Environment variables*.)

## Run
```bash
python main.py single "Camera is great but the battery is awful"
python main.py analyze                # all reviews in data/reviews.csv
python main.py analyze --limit 5      # quick demo
python main.py evaluate               # LLM vs lexicon baseline
python -m unittest discover tests     # unit tests
python main.py analyze --mock         # offline test without API key
```
Outputs are written to `outputs/` (`results.csv`, `report.json`, `evaluation.json`).
Use your own data: a CSV with a `review` column (and `label` column for `evaluate`).

## Design decisions (efficiency & quality)
- **Prompt engineering**: role + strict JSON schema + rules for negation/sarcasm +
  one few-shot example; `temperature: 0` for deterministic classification.
- **Robustness**: retry with exponential back-off on 429/5xx, JSON extraction from
  code fences, schema validation, one re-try on malformed output, baseline fallback.
- **Efficiency**: persistent session, on-disk cache (same review never re-billed),
  one report call per batch instead of per review, configurable request delay.
- **No hard-coded values**: model, prompts and paths all come from config files.
- `--mock` is only for testing the pipeline offline; its numbers are not real LLM results.

## Possible extensions
Streamlit UI, more languages, topic clustering of keywords, larger labelled dataset.

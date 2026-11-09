"""FR-5 — Sentiment analysis: one sentiment label and score for every cleaned review.

Model: cardiffnlp/twitter-xlm-roberta-base-sentiment (Hugging Face), a multilingual XLM-RoBERTa fine-tuned to
classify a text as negative / neutral / positive. It reads French, English, Korean, Spanish… without translation.

Input : data/processed/reviews_clean.csv          (output of step 01)
Output: data/processed/reviews_sentiment.csv      (same rows + 5 new columns, see below)
        data/processed/reviews_sentiment_report.json   (counts, and how well the labels agree with the star ratings)

New columns
  sentiment_label   positive / neutral / negative   (most likely class; empty when the review has no text)
  sentiment_binary  positive / negative              (FR-5 asks for positive or negative: sign of the score)
  sentiment_score   P(positive) - P(negative), from -1 (very negative) to +1 (very positive)
  p_negative, p_neutral, p_positive   the model's three probabilities

Usage (from the project folder, with the virtual environment active):
  python3 pipeline/05_review_sentiment.py              # all reviews (first run downloads the model, ~1.1 GB)
  python3 pipeline/05_review_sentiment.py --limit 200  # quick trial on 200 reviews
  python3 pipeline/05_review_sentiment.py --backend fake   # tests the code only, results are meaningless
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABELS = ["negative", "neutral", "positive"]


class RealModel:
    def __init__(self, batch_size: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {MODEL} on {self.device}…", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        # put the model's outputs in the order negative, neutral, positive whatever its internal order is
        self.order = [next(i for i, lab in id2label.items() if lab.startswith(name[:3])) for name in LABELS]
        self.batch_size = batch_size

    def predict(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            # long reviews are cut at 256 tokens (about 180 words): the start of a review carries most of its tone
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            out.append(probs[:, self.order])
            done = min(start + self.batch_size, len(texts))
            if (start // self.batch_size) % 20 == 0 or done == len(texts):
                print(f"  {done}/{len(texts)} reviews", flush=True)
        return np.vstack(out) if out else np.zeros((0, 3))


class FakeModel:
    """Deterministic stand-in to test the code without downloading anything. Its labels mean nothing."""

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.array([np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).dirichlet([1, 1, 1]) for t in texts])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only the first N reviews with text (quick trial)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backend", choices=["real", "fake"], default="real")
    parser.add_argument("--all-rows", action="store_true", help="also score reviews outside the study scope")
    args = parser.parse_args()

    started = time.time()
    src = ROOT / "data" / "processed" / "reviews_clean.csv"
    reviews = pd.read_csv(src, low_memory=False)
    if not args.all_rows and "in_scope" in reviews.columns:
        reviews = reviews[reviews["in_scope"]].copy()
    print(f"{len(reviews)} reviews read from {src.relative_to(ROOT)}")

    # the text the model reads: the original text (the model is multilingual, no translation needed)
    text = reviews["text"].fillna("").astype(str).str.strip()
    has_text = text.str.len() >= 3
    todo = reviews.index[has_text]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} reviews with text will be scored ({int((~has_text).sum())} have no text and stay empty)")

    model = FakeModel() if args.backend == "fake" else RealModel(args.batch_size)
    probs = model.predict(text.loc[todo].tolist())

    for col in ["sentiment_label", "sentiment_binary"]:
        reviews[col] = pd.Series(pd.NA, index=reviews.index, dtype="object")
    for col in ["sentiment_score", "p_negative", "p_neutral", "p_positive"]:
        reviews[col] = np.nan
    if len(todo):
        score = probs[:, 2] - probs[:, 0]
        reviews.loc[todo, "p_negative"] = probs[:, 0].round(4)
        reviews.loc[todo, "p_neutral"] = probs[:, 1].round(4)
        reviews.loc[todo, "p_positive"] = probs[:, 2].round(4)
        reviews.loc[todo, "sentiment_score"] = score.round(4)
        reviews.loc[todo, "sentiment_label"] = [LABELS[i] for i in probs.argmax(axis=1)]
        reviews.loc[todo, "sentiment_binary"] = np.where(score >= 0, "positive", "negative")

    out_dir = ROOT / "data" / "processed"
    target = out_dir / ("reviews_sentiment_FAKE.csv" if args.backend == "fake" else "reviews_sentiment.csv")
    reviews.to_csv(target, index=False, encoding="utf-8")

    # sanity check: does the model's label agree with the stars the reviewer gave?
    scored = reviews.loc[todo]
    stars = scored["rating"]
    star_class = np.where(stars >= 4, "positive", np.where(stars <= 2, "negative", "neutral"))
    clear = stars != 3
    agree_binary = float((scored.loc[clear, "sentiment_binary"].values == star_class[clear.values]).mean()) if clear.any() else None
    report = {
        "model": MODEL if args.backend == "real" else "FAKE (code test only)",
        "reviews_scored": int(len(todo)),
        "reviews_without_text": int((~has_text).sum()),
        "labels": {k: int(v) for k, v in scored["sentiment_label"].value_counts().items()},
        "binary": {k: int(v) for k, v in scored["sentiment_binary"].value_counts().items()},
        "mean_score_by_star": {str(int(k)): round(float(v), 3) for k, v in scored.groupby("rating")["sentiment_score"].mean().items()},
        "agreement_with_stars": {
            "rule": "1-2 stars = negative, 4-5 stars = positive (3-star reviews left out)",
            "binary_label_agrees": None if agree_binary is None else round(agree_binary, 3),
            "correlation_score_vs_stars": round(float(np.corrcoef(scored["sentiment_score"], stars)[0, 1]), 3) if len(scored) > 2 else None,
        },
        "seconds": round(time.time() - started),
    }
    (out_dir / "reviews_sentiment_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {target.relative_to(ROOT)}")
    print(f"Labels: {report['labels']}")
    print(f"Mean sentiment score by star rating: {report['mean_score_by_star']}")
    agree = report["agreement_with_stars"]["binary_label_agrees"]
    if agree is not None:
        print(f"The positive/negative label matches the stars for {agree:.1%} of 1-2 and 4-5 star reviews")
    print(f"Done in {report['seconds']} s")


if __name__ == "__main__":
    main()

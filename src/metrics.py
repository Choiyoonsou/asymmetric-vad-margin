import re

import numpy as np
from jiwer import cer, process_words


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text):
    if text is None or text != text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    normalized = text.strip().lower()
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized


def tokenize(text):
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def compute_text_metrics(reference, hypothesis):
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)

    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    first_word_match = False
    last_word_match = False
    if reference_tokens and hypothesis_tokens:
        first_word_match = reference_tokens[0] == hypothesis_tokens[0]
        last_word_match = reference_tokens[-1] == hypothesis_tokens[-1]

    return {
        "cer": cer(normalized_reference, normalized_hypothesis),
        "exact_match_rate": float(normalized_reference == normalized_hypothesis),
        "first_word_match_rate": float(first_word_match),
        "first_word_error_rate": float(not first_word_match),
        "last_word_match_rate": float(last_word_match),
        "last_word_error_rate": float(not last_word_match),
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
    }


def compute_word_csid(reference, hypothesis):
    output = process_words(normalize_text(reference), normalize_text(hypothesis))
    return {
        "word_c": float(output.hits),
        "word_s": float(output.substitutions),
        "word_d": float(output.deletions),
        "word_i": float(output.insertions),
    }


# A hypothesis longer than its reference can push CER above 1.0 without bound,
# and on attention backbones a handful of runaway utterances then dominate the
# mean. The follow-up review measured cases where mean and median disagree on
# the SIGN of a policy's effect, so every margin condition is reported on all
# three of these axes rather than the mean alone.
HALLUCINATION_CER_THRESHOLD = 1.0


def bootstrap_mean_ci(values, confidence=0.95, iterations=2000, seed=0):
    """Percentile bootstrap CI for the mean of a heavy-tailed CER sample."""
    values = np.asarray([v for v in values if v == v], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def summarize_cer(values, confidence=0.95, iterations=2000, seed=0):
    """Mean / median / capped mean / hallucination rate, plus a CI on the mean."""
    values = np.asarray([v for v in values if v == v], dtype=float)
    if values.size == 0:
        nan = float("nan")
        return {
            "cer_mean": nan,
            "cer_median": nan,
            "cer_capped_mean": nan,
            "hallucination_rate": nan,
            "cer_mean_ci_low": nan,
            "cer_mean_ci_high": nan,
            "n_examples": 0,
        }

    ci_low, ci_high = bootstrap_mean_ci(values, confidence, iterations, seed)
    return {
        "cer_mean": float(values.mean()),
        "cer_median": float(np.median(values)),
        "cer_capped_mean": float(np.minimum(values, HALLUCINATION_CER_THRESHOLD).mean()),
        "hallucination_rate": float((values > HALLUCINATION_CER_THRESHOLD).mean()),
        "cer_mean_ci_low": ci_low,
        "cer_mean_ci_high": ci_high,
        "n_examples": int(values.size),
    }

#!/usr/bin/env python3
"""train_gepa.py

Production-quality DSPy + GEPA pipeline for detecting *patient* emotional
frustration in healthcare call transcripts.

The pipeline:
  * Defines a strongly-instructed DSPy Signature (`DetectPatientFrustration`).
  * Wraps it in a `FrustrationClassifier` module (Chain-of-Thought).
  * Loads/validates a labeled CSV into DSPy Examples with a train/dev split.
  * Optimizes the prompt with DSPy's GEPA reflective optimizer, iteration by
    iteration, with early stopping on stalled F1.
  * Tracks prompt evolution (`prompt_history.json`, `best_prompt.txt`).
  * Produces evaluation metrics, an error-analysis CSV and a confusion matrix.

Designed to drive a *local* OpenAI-compatible model (e.g. "Moss 4B Thinking"
served on http://localhost:8000/v1).

Verified against the DSPy GEPA API as of DSPy 3.2.x (see README for version
assumptions).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import dspy
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "dspy is not installed. Run `pip install -r requirements.txt`."
    ) from exc

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Matplotlib uses a non-interactive backend so the script runs headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed every relevant RNG for reproducible splits and search."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger("gepa_frustration")


def configure_logging(verbosity: str = "INFO") -> None:
    level = getattr(logging, verbosity.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# --------------------------------------------------------------------------- #
# 1. DSPy Signature
# --------------------------------------------------------------------------- #
class DetectPatientFrustration(dspy.Signature):
    """Determine whether the PATIENT is expressing emotional frustration.

    You are analyzing a healthcare call transcript between two speakers:
      * "Caller" is the PATIENT.
      * "Callee" is the AGENT / receptionist.

    Decision target: classify ONLY the PATIENT'S (Caller's) emotional state.

    STRICT RULES
    ------------
    1. Focus only on the patient (Caller). Base the label exclusively on what
       the Caller says and how they say it.
    2. Ignore any frustration, sharpness or impatience expressed by the AGENT
       (Callee). The agent's speech is CONTEXT ONLY -- it never makes the label
       "yes" on its own.
    3. Decide using concrete linguistic evidence from the patient, such as:
         - emotional cues (anger, exasperation, sighing, "ugh", "seriously?")
         - explicit complaints about service, waiting, or treatment
         - repeated requests for the same thing (having to ask again)
         - negative sentiment directed at the practice or process
         - impatience ("how long is this going to take", "I've been on hold")
         - escalation attempts ("let me speak to a manager", "this is the
           third time I've called")
         - expressed dissatisfaction with outcomes or answers
    4. Do NOT guess and do NOT hallucinate evidence that is not in the text.
    5. Do NOT assume frustration when the evidence is weak. A patient who is
       merely assertive, direct, anxious about their health, or simply asking
       routine questions is NOT necessarily frustrated. When evidence is
       genuinely ambiguous or absent, prefer "no".

    OUTPUT
    ------
      - frustration: "yes" if the patient is frustrated, otherwise "no".
      - confidence: a calibrated probability in [0.0, 1.0] for your label.
      - reasoning: a short justification grounded in specific patient phrases.
    """

    transcript: str = dspy.InputField(
        desc="Full call transcript. Lines are prefixed 'Caller:' (patient) "
        "or 'Callee:' (agent)."
    )
    frustration: Literal["yes", "no"] = dspy.OutputField(
        desc="'yes' if the PATIENT (Caller) is frustrated, else 'no'."
    )
    confidence: float = dspy.OutputField(
        desc="Calibrated confidence for the label, between 0.0 and 1.0."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief justification citing specific patient phrasing/evidence."
    )


# --------------------------------------------------------------------------- #
# 3. DSPy Module
# --------------------------------------------------------------------------- #
class FrustrationClassifier(dspy.Module):
    """Chain-of-Thought classifier over `DetectPatientFrustration`.

    Chain-of-Thought is preferred over bare Predict here: frustration detection
    benefits from the model explicitly separating patient vs. agent speech and
    weighing evidence before committing to a label. GEPA then optimizes the
    underlying instructions (and the reasoning strategy) directly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(DetectPatientFrustration)

    def forward(self, transcript: str) -> dspy.Prediction:
        return self.classify(transcript=transcript)


# --------------------------------------------------------------------------- #
# 4. Dataset loader
# --------------------------------------------------------------------------- #
VALID_LABELS = {"yes", "no"}


def _normalize_label(raw: Any) -> str:
    label = str(raw).strip().lower()
    if label not in VALID_LABELS:
        raise ValueError(
            f"Invalid label {raw!r}. Labels must be one of {sorted(VALID_LABELS)}."
        )
    return label


def load_dataset(
    csv_path: str,
    train_ratio: float = 0.8,
    seed: int = SEED,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load and validate the CSV, returning (trainset, devset).

    The CSV must contain columns: call_id, transcript, label.
    Labels are validated and normalized to {"yes", "no"}. Each row becomes a
    `dspy.Example` whose only input is `transcript`. The split is stratified by
    label so both partitions keep a similar class balance, and is deterministic
    given `seed`.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"call_id": str})
    required = {"call_id", "transcript", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["transcript", "label"]).reset_index(drop=True)
    df["label"] = df["label"].apply(_normalize_label)

    if df.empty:
        raise ValueError("Dataset is empty after dropping invalid rows.")

    examples: list[dspy.Example] = []
    for _, row in df.iterrows():
        ex = dspy.Example(
            call_id=str(row["call_id"]),
            transcript=str(row["transcript"]),
            label=row["label"],
        ).with_inputs("transcript")
        examples.append(ex)

    # Deterministic, stratified split.
    rng = random.Random(seed)
    by_label: dict[str, list[dspy.Example]] = {"yes": [], "no": []}
    for ex in examples:
        by_label[ex.label].append(ex)

    trainset: list[dspy.Example] = []
    devset: list[dspy.Example] = []
    for label, group in by_label.items():
        rng.shuffle(group)
        cut = max(1, int(round(len(group) * train_ratio))) if len(group) > 1 else 1
        # Guarantee at least one dev example per class when possible.
        if len(group) > 1 and cut >= len(group):
            cut = len(group) - 1
        trainset.extend(group[:cut])
        devset.extend(group[cut:])

    rng.shuffle(trainset)
    rng.shuffle(devset)

    logger.info(
        "Loaded %d examples -> train=%d (yes=%d/no=%d), dev=%d (yes=%d/no=%d)",
        len(examples),
        len(trainset),
        sum(e.label == "yes" for e in trainset),
        sum(e.label == "no" for e in trainset),
        len(devset),
        sum(e.label == "yes" for e in devset),
        sum(e.label == "no" for e in devset),
    )
    if not devset:
        logger.warning("Dev split is empty; metrics will be computed on train.")
        devset = trainset
    return trainset, devset


# --------------------------------------------------------------------------- #
# 5. Evaluation metrics
# --------------------------------------------------------------------------- #
@dataclass
class PredictionRecord:
    call_id: str
    true_label: str
    predicted_label: str
    confidence: float
    transcript: str


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    records: list[PredictionRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _label_to_int(label: str) -> int:
    return 1 if label == "yes" else 0


def _safe_confidence(pred: dspy.Prediction) -> float:
    try:
        return float(getattr(pred, "confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def run_predictions(
    program: dspy.Module, dataset: list[dspy.Example]
) -> list[PredictionRecord]:
    """Run the program over a dataset, returning per-example records."""
    records: list[PredictionRecord] = []
    for ex in dataset:
        try:
            pred = program(transcript=ex.transcript)
            predicted = _normalize_label(getattr(pred, "frustration", "no"))
            confidence = _safe_confidence(pred)
        except Exception as exc:  # noqa: BLE001 - keep going on model errors
            logger.warning("Prediction failed for call_id=%s: %s", ex.call_id, exc)
            predicted, confidence = "no", 0.0
        records.append(
            PredictionRecord(
                call_id=ex.call_id,
                true_label=ex.label,
                predicted_label=predicted,
                confidence=confidence,
                transcript=ex.transcript,
            )
        )
    return records


def evaluate(program: dspy.Module, dataset: list[dspy.Example]) -> EvalResult:
    """Compute accuracy / precision / recall / F1 for the program on a set."""
    records = run_predictions(program, dataset)
    y_true = [_label_to_int(r.true_label) for r in records]
    y_pred = [_label_to_int(r.predicted_label) for r in records]

    return EvalResult(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        records=records,
    )


def log_metrics(title: str, result: EvalResult) -> None:
    logger.info("---- %s ----", title)
    logger.info("Accuracy : %.4f", result.accuracy)
    logger.info("Precision: %.4f", result.precision)
    logger.info("Recall   : %.4f", result.recall)
    logger.info("F1       : %.4f", result.f1)


# --------------------------------------------------------------------------- #
# 6. GEPA metric (per-example score + textual feedback)
# --------------------------------------------------------------------------- #
def frustration_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
):
    """GEPA feedback metric.

    GEPA evaluates one example at a time, so this returns per-example
    correctness (which the optimizer aggregates -- maximizing it drives F1 up).
    When called by GEPA (i.e. with a `pred_name`) it returns a
    `dspy.Prediction(score=..., feedback=...)` so the reflective optimizer can
    read *why* a prediction was right or wrong. When used as a plain metric
    (e.g. by `dspy.Evaluate`) it returns the bare float score.
    """
    try:
        predicted = _normalize_label(getattr(pred, "frustration", "no"))
    except Exception:  # noqa: BLE001
        predicted = "no"
    gold_label = gold.label
    correct = predicted == gold_label
    score = 1.0 if correct else 0.0

    if pred_name is None:
        # Plain metric usage (no feedback requested).
        return score

    reasoning = str(getattr(pred, "reasoning", "")).strip()
    if correct:
        feedback = (
            f"CORRECT. The true label is '{gold_label}' and the model predicted "
            f"'{predicted}'. Keep relying on concrete patient (Caller) evidence."
        )
    else:
        if gold_label == "yes":
            feedback = (
                f"WRONG. The true label is 'yes' (the PATIENT is frustrated) but "
                f"the model predicted 'no'. It likely MISSED patient frustration "
                f"signals such as complaints, repeated requests, impatience, "
                f"escalation attempts, or negative sentiment. Re-read only the "
                f"'Caller:' lines and look harder for these cues."
            )
        else:
            feedback = (
                f"WRONG. The true label is 'no' (the PATIENT is NOT frustrated) "
                f"but the model predicted 'yes'. It likely OVER-called "
                f"frustration -- possibly reacting to the AGENT's (Callee's) tone, "
                f"or treating routine/assertive/anxious patient speech as "
                f"frustration. Ignore the agent's words and require strong patient "
                f"evidence before answering 'yes'."
            )
        if reasoning:
            feedback += f" Model's stated reasoning was: \"{reasoning[:300]}\"."

    return dspy.Prediction(score=score, feedback=feedback)


# --------------------------------------------------------------------------- #
# 7 & 10. Optimization loop with prompt tracking and early stopping
# --------------------------------------------------------------------------- #
def extract_prompt(program: dspy.Module) -> str:
    """Serialize the current instructions of every predictor in the program."""
    chunks: list[str] = []
    for name, predictor in program.named_predictors():
        sig = getattr(predictor, "signature", None)
        instructions = getattr(sig, "instructions", "") if sig else ""
        chunks.append(f"### predictor: {name}\n{instructions}")
    return "\n\n".join(chunks).strip()


def build_lm(model: str, api_base: str, api_key: str, max_tokens: int) -> dspy.LM:
    """Construct a DSPy LM pointing at an OpenAI-compatible endpoint."""
    model_id = model if "/" in model else f"openai/{model}"
    return dspy.LM(
        model=model_id,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0,
        max_tokens=max_tokens,
        model_type="chat",
    )


def optimize(
    student: dspy.Module,
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    reflection_lm: dspy.LM,
    args: argparse.Namespace,
) -> tuple[dspy.Module, list[dict[str, Any]]]:
    """Run GEPA iteratively with prompt tracking and early stopping.

    Each outer "iteration" is a GEPA optimization round with a bounded metric-
    call budget. The best program is carried forward as the student for the
    next round, giving us iteration-level control to (a) snapshot the prompt and
    its dev F1 and (b) apply the early-stopping rule:

        stop if F1 improvement < `min_delta` for `patience` consecutive rounds.
    """
    history: list[dict[str, Any]] = []

    # Iteration 0 = the strong initial prompt, before any optimization.
    baseline = evaluate(student, devset)
    log_metrics("Iteration 0 (initial prompt) - dev metrics", baseline)
    history.append(
        {
            "iteration": 0,
            "score": round(baseline.f1, 6),
            "metrics": {k: round(v, 6) for k, v in baseline.as_dict().items()},
            "prompt": extract_prompt(student),
        }
    )

    best_program = student
    best_f1 = baseline.f1
    stalled = 0

    for iteration in range(1, args.max_iters + 1):
        logger.info(
            "=== GEPA iteration %d/%d (budget=%d metric calls) ===",
            iteration,
            args.max_iters,
            args.metric_calls_per_iter,
        )

        gepa = dspy.GEPA(
            metric=frustration_metric,
            max_metric_calls=args.metric_calls_per_iter,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=args.reflection_minibatch_size,
            candidate_selection_strategy="pareto",
            num_threads=args.num_threads,
            track_stats=True,
            seed=args.seed + iteration,
            log_dir=os.path.join(args.output_dir, f"gepa_logs_iter{iteration}"),
        )

        try:
            candidate = gepa.compile(
                best_program, trainset=trainset, valset=devset
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("GEPA iteration %d failed: %s", iteration, exc)
            logger.error("Stopping optimization; keeping best program so far.")
            break

        result = evaluate(candidate, devset)
        log_metrics(f"Iteration {iteration} - dev metrics", result)

        history.append(
            {
                "iteration": iteration,
                "score": round(result.f1, 6),
                "metrics": {k: round(v, 6) for k, v in result.as_dict().items()},
                "prompt": extract_prompt(candidate),
            }
        )
        save_prompt_history(history, args.output_dir)  # persist after each round

        improvement = result.f1 - best_f1
        if result.f1 > best_f1:
            logger.info(
                "New best F1: %.4f (was %.4f, +%.4f)",
                result.f1,
                best_f1,
                improvement,
            )
            best_program = candidate
            best_f1 = result.f1

        # 10. Early stopping on stalled F1.
        if improvement < args.min_delta:
            stalled += 1
            logger.info(
                "F1 improvement %.4f < min_delta %.4f (stalled %d/%d).",
                improvement,
                args.min_delta,
                stalled,
                args.patience,
            )
        else:
            stalled = 0

        if stalled >= args.patience:
            logger.info(
                "Early stopping: no F1 gain >= %.4f for %d consecutive iterations.",
                args.min_delta,
                args.patience,
            )
            break

    return best_program, history


# --------------------------------------------------------------------------- #
# Artifact writers
# --------------------------------------------------------------------------- #
def save_prompt_history(history: list[dict[str, Any]], output_dir: str) -> None:
    path = os.path.join(output_dir, "prompt_history.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)
    logger.debug("Wrote prompt history -> %s", path)


def save_best_prompt(program: dspy.Module, output_dir: str) -> None:
    path = os.path.join(output_dir, "best_prompt.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(extract_prompt(program))
    logger.info("Best prompt saved -> %s", path)


def save_program(program: dspy.Module, output_dir: str) -> None:
    """Persist the optimized program so it can be reloaded for inference."""
    path = os.path.join(output_dir, "best_program.json")
    try:
        program.save(path)
        logger.info("Optimized program saved -> %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save program state: %s", exc)


def save_error_analysis(result: EvalResult, output_dir: str, snippet_len: int = 280) -> None:
    """8. Write one row per misclassified example."""
    path = os.path.join(output_dir, "error_analysis.csv")
    rows = []
    for r in result.records:
        if r.predicted_label != r.true_label:
            snippet = " ".join(r.transcript.split())[:snippet_len]
            rows.append(
                {
                    "call_id": r.call_id,
                    "true_label": r.true_label,
                    "predicted_label": r.predicted_label,
                    "confidence": round(r.confidence, 4),
                    "transcript_snippet": snippet,
                }
            )
    pd.DataFrame(
        rows,
        columns=[
            "call_id",
            "true_label",
            "predicted_label",
            "confidence",
            "transcript_snippet",
        ],
    ).to_csv(path, index=False)
    logger.info("Error analysis (%d mistakes) saved -> %s", len(rows), path)


def save_confusion_matrix(result: EvalResult, output_dir: str) -> None:
    """9. Render and save a confusion-matrix PNG."""
    path = os.path.join(output_dir, "confusion_matrix.png")
    labels = ["no", "yes"]
    y_true = [_label_to_int(r.true_label) for r in result.records]
    y_pred = [_label_to_int(r.predicted_label) for r in result.records]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Patient Frustration - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)

    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", path)


# --------------------------------------------------------------------------- #
# 11 & 12. CLI / configuration
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DSPy + GEPA prompt optimization for patient frustration "
        "detection in healthcare call transcripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--dataset", required=True, help="Path to labeled CSV.")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction.")
    p.add_argument("--output-dir", default="artifacts", help="Where to write outputs.")

    # Task model (the local LLM being optimized)
    p.add_argument("--model", default="moss", help="Model name served by the endpoint.")
    p.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL for the task model.",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "local-key"),
        help="API key (often a dummy value for local servers).",
    )
    p.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per call.")

    # Reflection model (used by GEPA to propose new prompts; a strong model is
    # recommended). Defaults to the same endpoint/model as the task model.
    p.add_argument(
        "--reflection-model",
        default=None,
        help="Model for GEPA reflection (defaults to --model).",
    )
    p.add_argument(
        "--reflection-api-base",
        default=None,
        help="API base for the reflection model (defaults to --api-base).",
    )
    p.add_argument(
        "--reflection-max-tokens",
        type=int,
        default=8192,
        help="Max tokens for reflection generations (prompts can be long).",
    )

    # GEPA / optimization
    p.add_argument("--max-iters", type=int, default=20, help="Max GEPA rounds.")
    p.add_argument(
        "--metric-calls-per-iter",
        type=int,
        default=60,
        help="GEPA metric-call budget per optimization round.",
    )
    p.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=3,
        help="Examples reflected on per GEPA proposal step.",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Parallel threads for GEPA evaluation (default: library default).",
    )

    # Early stopping
    p.add_argument(
        "--min-delta",
        type=float,
        default=0.01,
        help="Minimum F1 improvement to count as progress.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many consecutive rounds below --min-delta.",
    )

    # Misc
    p.add_argument("--seed", type=int, default=SEED, help="Global random seed.")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Evaluate the initial prompt only (no GEPA run).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Configuration: %s", vars(args))

    # Configure the task LM.
    task_lm = build_lm(args.model, args.api_base, args.api_key, args.max_tokens)
    dspy.configure(lm=task_lm)
    logger.info("Task model configured: %s @ %s", args.model, args.api_base)

    # Configure the reflection LM used by GEPA.
    reflection_lm = build_lm(
        args.reflection_model or args.model,
        args.reflection_api_base or args.api_base,
        args.api_key,
        args.reflection_max_tokens,
    )
    # Reflection benefits from sampling diversity.
    reflection_lm.kwargs["temperature"] = 1.0

    # Load data.
    trainset, devset = load_dataset(args.dataset, args.train_ratio, args.seed)

    # Build the initial (strongly-instructed) program.
    program = FrustrationClassifier()

    if args.skip_optimization:
        logger.info("Skipping optimization (--skip-optimization).")
        best_program = program
        history = [
            {
                "iteration": 0,
                "score": round(evaluate(program, devset).f1, 6),
                "prompt": extract_prompt(program),
            }
        ]
    else:
        best_program, history = optimize(
            program, trainset, devset, reflection_lm, args
        )

    # Persist prompt evolution artifacts.
    save_prompt_history(history, args.output_dir)
    save_best_prompt(best_program, args.output_dir)
    save_program(best_program, args.output_dir)

    # Final evaluation + analysis artifacts on the dev set.
    final = evaluate(best_program, devset)
    log_metrics("FINAL - best program, dev metrics", final)
    save_error_analysis(final, args.output_dir)
    save_confusion_matrix(final, args.output_dir)

    logger.info("Best dev F1: %.4f", final.f1)
    logger.info("All artifacts written to: %s", os.path.abspath(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

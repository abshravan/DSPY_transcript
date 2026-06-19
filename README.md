# Patient Frustration Detection — DSPy + GEPA Pipeline

A production-quality [DSPy](https://dspy.ai) pipeline that uses the **GEPA**
reflective prompt optimizer to automatically improve prompting for **emotion
(frustration) detection in healthcare call transcripts**.

Each transcript has two speakers:

- `Caller` → the **patient**
- `Callee` → the **agent / receptionist**

The task is binary: is the **patient** expressing emotional frustration?
(`yes` / `no`). Only the patient's speech drives the label; the agent's speech
is context only.

The pipeline targets a **local, OpenAI-compatible LLM** (e.g. *Moss 4B
Thinking*) served at a URL such as `http://localhost:8000/v1`.

---

## What the pipeline does

1. **DSPy Signature** — `DetectPatientFrustration` with input `transcript` and
   outputs `frustration: Literal["yes","no"]`, `confidence: float`,
   `reasoning: str`.
2. **Strong initial prompt** — a detailed instruction set (focus on the patient,
   ignore the agent, use linguistic/emotional/complaint/escalation evidence,
   avoid guessing/hallucinating, prefer `no` when evidence is weak).
3. **Module** — `FrustrationClassifier(dspy.Module)` built on
   `dspy.ChainOfThought` (CoT helps the model separate speakers and weigh
   evidence before committing to a label).
4. **Dataset loader** — `load_dataset(csv_path)` validates labels, builds DSPy
   `Example`s, and produces a deterministic, **stratified** train/dev split.
5. **Metrics** — accuracy, precision, recall, F1 (scikit-learn). Printed/logged
   after every optimization round.
6. **GEPA optimization** — `dspy.GEPA` optimizes instructions, reasoning
   strategy, and output-format guidance over multiple iterations using a
   per-example **feedback** metric.
7. **Prompt evolution tracking** — `prompt_history.json` (iteration, score,
   prompt) and `best_prompt.txt`.
8. **Error analysis** — `error_analysis.csv` (`call_id`, `true_label`,
   `predicted_label`, `confidence`, `transcript_snippet`) for every mistake.
9. **Confusion matrix** — `confusion_matrix.png` (matplotlib).
10. **Early stopping** — stop when F1 improvement `< 0.01` for 3 consecutive
    iterations (configurable via `--min-delta` / `--patience`).
11. **Local model support** — `dspy.LM` against any OpenAI-compatible
    `--api-base`, fully configured via `argparse`.
12. **CLI**, **reproducible seeds** (`random`, `numpy`, DSPy/GEPA `seed`), and
    structured **logging** throughout.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Requires Python 3.10+ (uses `X | Y` type syntax).

---

## Serving the local model

Any OpenAI-compatible server works (vLLM, llama.cpp server, LM Studio, Ollama's
OpenAI shim, etc.). For example, with vLLM:

```bash
vllm serve moss-4b-thinking --port 8000
# -> exposes http://localhost:8000/v1
```

The pipeline talks to it through DSPy's LiteLLM-backed `dspy.LM`. Model names
are prefixed with `openai/` automatically if you don't include a provider.

---

## Usage

```bash
python train_gepa.py \
    --dataset sample_data.csv \
    --model moss \
    --api-base http://localhost:8000/v1 \
    --train-ratio 0.8 \
    --max-iters 20
```

Quickly check the **initial** prompt without running GEPA:

```bash
python train_gepa.py --dataset sample_data.csv --skip-optimization
```

### Key options

| Flag | Default | Purpose |
|------|---------|---------|
| `--dataset` | (required) | Path to the labeled CSV. |
| `--model` | `moss` | Model name served by the endpoint. |
| `--api-base` | `http://localhost:8000/v1` | OpenAI-compatible endpoint for the task model. |
| `--api-key` | `$OPENAI_API_KEY` or `local-key` | API key (dummy is fine for local). |
| `--train-ratio` | `0.8` | Train fraction (stratified split). |
| `--max-iters` | `20` | Max GEPA optimization rounds. |
| `--metric-calls-per-iter` | `60` | GEPA metric-call budget per round. |
| `--reflection-model` / `--reflection-api-base` | `--model` / `--api-base` | LM GEPA uses to *propose* new prompts. A stronger model here usually improves results. |
| `--min-delta` | `0.01` | Min F1 gain counted as progress. |
| `--patience` | `3` | Consecutive stalled rounds before early stop. |
| `--num-threads` | library default | Parallelism for GEPA evaluation. |
| `--seed` | `42` | Global RNG seed. |
| `--output-dir` | `artifacts` | Where all artifacts are written. |

Run `python train_gepa.py --help` for the full list.

---

## Input format

```csv
call_id,transcript,label
123,"Caller: ...\nCallee: ...",yes
124,"Caller: ...\nCallee: ...",no
```

- `label` ∈ {`yes`, `no`} (case-insensitive; validated on load).
- Transcript lines should be prefixed with `Caller:` (patient) / `Callee:`
  (agent). A ready-to-run `sample_data.csv` is included.

---

## Outputs (written to `--output-dir`, default `artifacts/`)

| File | Description |
|------|-------------|
| `prompt_history.json` | `[{iteration, score, metrics, prompt}, ...]` per round. |
| `best_prompt.txt` | The best optimized prompt/instructions. |
| `best_program.json` | Serialized optimized DSPy program (reloadable). |
| `error_analysis.csv` | One row per misclassification. |
| `confusion_matrix.png` | Confusion matrix for the best program on dev. |
| `gepa_logs_iter*/` | GEPA's own per-round logs. |

---

## How GEPA is used (and version notes)

- Verified against the DSPy **GEPA** API in **DSPy 3.x** (latest: 3.2.1).
  Constructor used:
  `dspy.GEPA(metric, max_metric_calls=..., reflection_lm=...,
  reflection_minibatch_size=..., candidate_selection_strategy="pareto",
  track_stats=True, seed=..., log_dir=...)` and
  `compile(student, trainset=..., valset=...)`.
- The metric (`frustration_metric`) follows GEPA's
  `(gold, pred, trace, pred_name, pred_trace)` signature and returns a
  `dspy.Prediction(score=..., feedback=...)` so the reflective optimizer can see
  *why* each prediction was right or wrong. It also works as a plain float
  metric when `pred_name` is `None` (e.g. for `dspy.Evaluate`).
- **Iteration-level control / early stopping:** GEPA manages its own internal
  search budget, so to implement the requested per-iteration early-stopping rule
  this pipeline runs GEPA in **bounded rounds** (`--metric-calls-per-iter`),
  carrying the best program forward as the student each round, snapshotting the
  prompt + dev F1, and stopping when F1 stalls. This is a thin wrapper over the
  public API — no GEPA internals are touched.
- GEPA optimizes a *per-example* objective; corpus-level **F1/precision/recall**
  are computed separately (scikit-learn) for reporting and the early-stop rule.

### Version assumptions

- `dspy>=3.0,<4.0` (pinned in `requirements.txt`). On DSPy 2.6.x the GEPA API is
  similar but may differ slightly; if you must use 2.6.x, confirm the `GEPA`
  constructor/`compile` signatures match those above.
- A strong reflection model materially improves GEPA. With only a small local
  model available, point `--reflection-model` / `--reflection-api-base` at the
  strongest endpoint you can.

---

## Reproducibility

`random`, `numpy`, and DSPy/GEPA seeds are all set from `--seed` (default `42`).
The train/dev split is deterministic and stratified. Set task-model
`temperature=0.0` (done by default) for stable evaluation.

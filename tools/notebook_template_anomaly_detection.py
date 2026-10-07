"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

This is the anomaly-scoring tutorial of moment-pipeline; the repository ships three TASK-INFERENCE notebooks
(embeddings, imputation, anomaly scoring) generated from three templates that share the same carried
package (ten modules under src/moment_pipeline/, in dependency order) and the same model cell.

Generator /2 keys in use: ``modules`` lists every module except ``__init__.py``; ``entry_module`` is
``model.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/``MODEL_LICENSE``/``MODEL_KEY``); ``model_load`` is
the package's documented loader ``load_moment(task="reconstruction", weights_dir=WEIGHTS_DIR)`` — MOMENT
replaces its head per task (RFC M-8), so the task is part of the loaded model's identity.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    'package': 'moment_pipeline',
    'repo_name': 'moment-pipeline',
    'stem': 'moment_anomaly_detection',
    'notebook_name': 'moment_anomaly_detection_colab.ipynb',
    'profile': 'TASK-INFERENCE',
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    # MAD-m5: the measured size of the locked wheels (sum of the locked manylinux x86_64 wheel sizes from PyPI
    # release metadata, read on 2026-10-07; the lock's torch 2.14.0 is the CUDA build and carries the NVIDIA libraries).
    "lock_download": {"bytes": 3078108227, "measured": "sum of the locked manylinux x86_64 wheel sizes from PyPI release metadata, 2026-10-07; the lock's `torch==2.14.0` is the CUDA build, so the CPU path downloads its NVIDIA libraries too"},
    'pipeline_class': 'LoadedMoment',
    'weights_key': 'moment-1-base',
    'modules': ['anomaly.py', 'canonical.py', 'config.py', 'csvio.py', 'embedding.py', 'imputation.py', 'model.py', 'provenance.py', 'roles.py', 'validation.py'],
    'entry_module': 'model.py',
    'model_load': 'load_moment(task="reconstruction", weights_dir=WEIGHTS_DIR)',
    'runtime_imports': ['torch', 'transformers', 'pandas', 'numpy'],
    'title': 'MOMENT anomaly scoring — DIMER task-inference tutorial (standalone)',
    'badges': [
        (
            'GitHub',
            'https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/kurtvalcorza/moment-pipeline',
        ),
        (
            'Open In Colab',
            'https://colab.research.google.com/assets/colab-badge.svg',
            'https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_anomaly_detection_colab.ipynb',
        ),
        (
            'Hugging Face',
            'https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutonLab%2FMOMENT--1--base-ffcc4d?style=flat',
            'https://huggingface.co/AutonLab/MOMENT-1-base',
        ),
        (
            'Upstream',
            'https://img.shields.io/badge/Upstream-moment--timeseries--foundation--model%2Fmoment-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/moment-timeseries-foundation-model/moment',
        ),
        (
            'arXiv',
            'https://img.shields.io/badge/arXiv-2402.03885-b31b1b.svg',
            'https://arxiv.org/abs/2402.03885',
        ),
    ],
    'capability': (
        'raw reconstruction-residual anomaly ranking (no threshold, no binary detector) with the pinned `AutonLab/MOMENT-1-base` checkpoint'
    ),
    'intro': (
        "This notebook demonstrates **raw reconstruction-residual scoring**, not a binary detector, through the repository's public `moment_pipeline` API carried in this notebook. MOMENT sees each scored point and the repository reports its self-reconstruction residual. Under the default MAE rule, **higher residual scores mean stronger anomaly evidence according to this score**. There is **no universal/default threshold in v1** and this tutorial never converts scores into binary anomaly labels. **No gradient training, fine-tuning, in-context conditioning, or fitted preprocessing occurs** — **no adaptation occurs.** **Upstream vs. this repository.** Upstream MOMENT supplies the pretrained reconstruction model. This repository supplies immutable pinning/integrity verification, long-format validation and canonicalization, explicit residual/aggregation policies, scored-domain accounting, threshold non-policy, the `top_k_recall` ranking check, and machine-readable provenance."
    ),
    "guided": {"opening": [(
        '**Who this notebook is for.** A learner who knows basic Python and pandas, has used Colab or Jupyter and has met time-series data, and wants to see how a pretrained time-series foundation model is used without any training to rank unusual points — what goes in, what comes out, and what the output does and does not prove. The audience is students and practitioners preparing their own sensor or monitoring series; no prior experience with MOMENT is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | a long-format CSV: the synthetic series `A` with three injected `vibration` spikes (and their labels) or your own unlabelled CSV |\n| Model | the MOMENT-1-base reconstruction head (`task="reconstruction"`); each point is scored by how badly the model reconstructs it |\n| Output | a raw, uncalibrated residual score per series, channel and timestamp (higher = stronger anomaly evidence **within one channel of one series**; raw residuals are not comparable across channels), per-channel residual statistics, a top-k recall check on the labelled sample beside a naive |z-score| baseline and the pooled all-channel ranking, and no threshold |\n\n**How to use this notebook.** Choose a runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook\'s own Python, so no restart is needed (the recorded hosted runs of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the default path. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer. No per-cell outputs of the hosted runs are recorded in this repository (only their pass/fail), so the worked answers state what the code and the sample guarantee rather than quoting numbers; compare them with what your run prints. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the synthetic sample with injected spikes (or your CSV) → 5 validation and canonicalization *(core concept: windows, padding, masks)* → 6 raw residual scores and their per-channel scale *(core concept: a score is not a decision, and is comparable only within a channel)* → 7 ranking, top-k recall, a naive baseline and the pooled ranking *(evaluation practice)* → 8 the plot over time, per series *(the spikes marked)* → 9 → export and provenance *(engineering)* → conclude.'
    )]},
    'learning_objectives': (
        'install the pinned runtime; read what the carried package guarantees; resolve and digest-verify the immutable model revision; generate the deterministic labelled synthetic sample or bring your own long-format CSV; validate and canonicalize the input into an input manifest; compute raw anomaly scores through the production-facing API; interpret score direction and threshold semantics; check the injected-spike ranking quantitatively through an evaluation report that is `sample-sanity` only when labels exist; and export raw scores with provenance. By the end of this notebook you will be able to do each of these without the repository being reachable.'
    ),
    'exclusions': (
        'a calibrated detector, a universal threshold, forecasting, classification, or production fitness. High reconstruction error and real-world anomaly status are not equivalent concepts.'
    ),
    'prerequisites': [
        '- **Learner:** basic Python and pandas familiarity; no prior experience with MOMENT or time-series foundation models. Windows, patches, padding, masks and the evaluation verdicts are explained where they are first used and again in the Glossary.',
        '- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels (plus `momentfm`, built from its pinned commit), so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported.',
        '- **Compute:** CPU is the default path and no GPU is required; CUDA is used automatically when available. The public v1 API accepts `float32` only. The locked `torch==2.14.0` wheel (the CUDA build, ~555 MB) and its NVIDIA libraries are the largest downloads of the run — see the measured total under External access — followed by the ~454 MB `model.safetensors`.',
        '- **Knowledge:** basic Python and pandas; what a long-format time-series table is.',
        "- **Data:** the default sample is the repository's deterministic two-channel synthetic series with three documented injected spikes in the `vibration` channel, regenerated in code together with its label table, so nothing is downloaded and no private data is needed. Labels make ranking behaviour falsifiable; they are not calibration data or benchmark evidence. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 CSV with columns `series_id`, `timestamp`, `channel`, `value`; `value` numeric or missing; identifiers and timestamps valid. Duplicate or ambiguous column names are rejected from the raw CSV header before dataframe parsing, and duplicate `(series_id, channel, timestamp)` rows are rejected by production validation. BYOD does not require anomaly labels; without labels the notebook ranks residuals but cannot measure detector quality. Operational ceilings: at most 5,000,000 rows, 1,024 series, 32 channels, and 1,024 canonical windows; MOMENT uses 512-step windows and 8-step non-overlapping patches; short series are left-padded, long series keep the final 512 timestamps, irregular spacing is surfaced rather than silently interpolated. BYOD is read locally in the notebook runtime and is not sent to an external inference service. Do not upload confidential or restricted data (personal or otherwise sensitive data included) to a hosted notebook environment unless you are authorized to do so.",
    ],
    "cells": [
        {
            "md": (
                '## 4. Generate the synthetic sample or optional BYOD\n'
                '\n'
                "The default sample is **synthetic**: the repository's `examples/sample-data/generate_samples.py` formulas (one series `A`, channels `vibration` and `temperature`, 256 steps at 15 minutes — a trend plus two sinusoids) regenerated in code and rendered to the same canonical CSV bytes, so the `moment_anomaly.csv` SHA-256 is asserted against the digest the repository checks in. The three injected spikes in the `vibration` channel come with a label table, digest-asserted the same way, so the ranking check later is falsifiable; BYOD carries no labels. It is deterministic teaching data, not benchmark evidence. The optional upload path first validates the **raw CSV header** with `read_long_csv_bytes()` (duplicate or ambiguous names cannot be silently renamed by pandas); the resulting frame still goes through the same production validation/canonicalization path in the next stage. Set `USE_BYOD=True` and `BYOD_PATH` to a CSV already in the runtime (Colab, Kaggle or Jupyter); on Colab an empty `BYOD_PATH` opens an upload dialog, and a cancelled upload stops with a message. `DIMER_BYOD_PATH` remains the automation hook. Successful completion means one identified input frame is available and its source/digest are recorded; look for the sample identity (kind, name, digest) and the first rows."
            ),
            "code": (
                'import hashlib\n'
                'import io\n'
                'import json\n'
                'from pathlib import Path\n'
                '\n'
                'import numpy as np\n'
                'import pandas as pd\n'
                '\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = ""  # @param {{type:"string"}}\n'
                '# The form path is used when USE_BYOD is on; DIMER_BYOD_PATH remains the automation hook.\n'
                'BYOD_PATH = (BYOD_PATH.strip() if USE_BYOD else "") or os.environ.get("DIMER_BYOD_PATH", "")\n'
                'SAMPLE_SHA256 = "58855d8961c2b0547e27763ffc95976435180477c2d31f0e217f64bb6dd19c54"  # examples/sample-data/SHA256SUMS\n'
                'LABELS_SHA256 = "f35ec637441c72ee6a16d396c2e76b24de8a19c17ba62e415aefe1bfdc155f3e"  # examples/sample-data/SHA256SUMS\n'
                '\n'
                '\n'
                'def build_samples() -> dict:\n'
                "    # The repository's examples/sample-data/generate_samples.py formulas; no random state.\n"
                '    n = 256\n'
                '    timestamps = pd.date_range("2026-01-01", periods=n, freq="15min")\n'
                '    step = np.arange(n, dtype=float)\n'
                '    rows = []\n'
                '    for channel, base, amplitude, period, phase, trend in (("vibration", 1.5, 0.35, 32.0, 0.0, 0.0008), ("temperature", 28.0, 2.5, 96.0, 9.0, 0.0015)):\n'
                '        values = base + trend * step + amplitude * np.sin(2.0 * np.pi * (step + phase) / period) + 0.08 * np.cos(2.0 * np.pi * step / 16.0)\n'
                '        rows.extend(("A", stamp, channel, float(value)) for stamp, value in zip(timestamps, values, strict=True))\n'
                '    clean = pd.DataFrame(rows, columns=["series_id", "timestamp", "channel", "value"])\n'
                '    anomaly = clean.copy()\n'
                '    labels = pd.DataFrame({{"series_id": "A", "timestamp": timestamps, "is_injected_anomaly": False}})\n'
                '    for index, delta in {{184: 2.8, 201: -3.2, 233: 3.6}}.items():\n'
                '        labels.loc[index, "is_injected_anomaly"] = True\n'
                '        selector = (anomaly["channel"] == "vibration") & (anomaly["timestamp"] == timestamps[index])\n'
                '        anomaly.loc[selector, "value"] = anomaly.loc[selector, "value"] + delta\n'
                '    return {{"moment_clean.csv": clean, "moment_anomaly.csv": anomaly, "moment_anomaly_labels.csv": labels}}\n'
                '\n'
                '\n'
                'def sample_csv_bytes(frame: pd.DataFrame) -> bytes:\n'
                '    return frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.6f", lineterminator="\\n").encode("utf-8")\n'
                '\n'
                '\n'
                'if BYOD_PATH:\n'
                '    if not Path(BYOD_PATH).expanduser().is_file():\n'
                '        raise FileNotFoundError(f"BYOD_PATH {{BYOD_PATH!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one long-format CSV.")\n'
                '    payload = Path(BYOD_PATH).expanduser().read_bytes()\n'
                '    frame = read_long_csv_bytes(payload)\n'
                '    labels = None  # BYOD carries no anomaly labels; the ranking is shown without a correctness metric\n'
                '    sample_identity = {{"kind": "byod", "name": Path(BYOD_PATH).name, "sha256": hashlib.sha256(payload).hexdigest()}}\n'
                '    sample_kind = "BYOD"\n'
                'elif USE_BYOD:\n'
                '    try:\n'
                '        from google.colab import files\n'
                '    except ImportError:\n'
                '        raise RuntimeError("USE_BYOD is on but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: copy the CSV into this runtime (or attach it as a Kaggle dataset) and set BYOD_PATH.") from None\n'
                '    uploaded = files.upload()\n'
                '    if len(uploaded) != 1:\n'
                '        raise ValueError(f"Upload exactly one CSV with columns series_id,timestamp,channel,value (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.")\n'
                '    name, payload = next(iter(uploaded.items()))\n'
                '    frame = read_long_csv_bytes(payload)\n'
                '    labels = None  # BYOD carries no anomaly labels; the ranking is shown without a correctness metric\n'
                '    sample_identity = {{"kind": "byod", "name": name, "sha256": hashlib.sha256(payload).hexdigest()}}\n'
                '    sample_kind = "BYOD"\n'
                'else:\n'
                '    samples = build_samples()\n'
                '    payload = sample_csv_bytes(samples["moment_anomaly.csv"])\n'
                '    observed = hashlib.sha256(payload).hexdigest()\n'
                '    if observed != SAMPLE_SHA256:\n'
                '        raise ValueError(f"Synthetic sample digest mismatch: {{observed}} != {{SAMPLE_SHA256}}")\n'
                '    frame = read_long_csv_bytes(payload)\n'
                '    sample_identity = {{"kind": "synthetic", "name": "moment_anomaly.csv", "sha256": observed}}\n'
                '    labels = samples["moment_anomaly_labels.csv"]\n'
                '    label_payload = sample_csv_bytes(labels)\n'
                '    label_digest = hashlib.sha256(label_payload).hexdigest()\n'
                '    if label_digest != LABELS_SHA256:\n'
                '        raise ValueError(f"Synthetic label digest mismatch: {{label_digest}} != {{LABELS_SHA256}}")\n'
                '    labels["timestamp"] = pd.to_datetime(labels["timestamp"])\n'
                '    sample_identity["labels_name"] = "moment_anomaly_labels.csv"\n'
                '    sample_identity["labels_sha256"] = label_digest\n'
                '    sample_kind = "synthetic"\n'
                '\n'
                'print({{"sample_kind": sample_kind, **sample_identity, "rows": len(frame), "columns": list(frame.columns)}})\n'
                'print(frame.head())'
            ),
        },
        {
            "md": (
                '## 5. Validate and canonicalize → input manifest\n'
                '\n'
                "Before the model runs, the cell prints the effective runtime and the operational ceilings — the `ResourceLimits` (rows, series, channels, windows), the fixed 512-step window and 8-step patch — and the device/precision policy. `validate_inputs` is the package's public validation stage: it runs exactly the two calls every task path makes (`validate_long_frame`, then `to_windows`), so it raises exactly what canonicalization would raise, and returns an **input manifest** naming the schema and ceilings, each window's series, valid positions, padding and truncation, the source-missingness fractions, and the verdict; it is written to `outputs/moment_anomaly_detection_input_manifest.json`. To show what rejection looks like, the cell also validates a deliberately broken copy (a channel with no observed value) and records the pipeline's own error code as a finding. The canonical `WindowSet` used by the model is built by the same calls; any padding, truncation, source missingness, or irregular frequency is disclosed here before model execution. Successful output means the input passed validation and every canonicalization effect is disclosed before the model runs."
            ),
            "code": (
                'import os\n'
                '\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                'config = MomentConfig(task="reconstruction")\n'
                'limits = config.limits\n'
                'print({{"python": platform.python_version(), "torch": torch.__version__, "device": pipe.identity.device, "dtype": pipe.identity.dtype}})\n'
                'print({{"ceilings": {{"max_rows": limits.max_rows, "max_series": limits.max_series, "max_channels": limits.max_channels, "max_windows": limits.max_windows, "sequence_length": config.sequence_length, "patch_length": config.patch_length}}}})\n'
                '\n'
                'report, normalized = validate_long_frame(frame, config)\n'
                'windows = to_windows(normalized, config, report=report, frame=normalized)\n'
                'input_manifest = validate_inputs(frame, config, names=[str(w) for w in windows.window_ids])\n'
                '# Demonstrate rejection on an input that breaks a rule; the finding is recorded, not swallowed.\n'
                'broken = frame.copy()\n'
                'broken.loc[broken["channel"] == broken["channel"].iloc[0], "value"] = np.nan\n'
                'try:\n'
                '    validate_inputs(broken, config)\n'
                'except ValidationError as exc:\n'
                '    input_manifest["findings"].append({{"input": "empty-channel-probe", "verdict": "rejected", "code": exc.code, "message": str(exc)}})\n'
                'with open("outputs/moment_anomaly_detection_input_manifest.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(input_manifest, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(input_manifest, indent=2, default=str))\n'
                'print("window tensor:", windows.x_enc.shape)\n'
                'print("padded windows:", int(sum(windows.padded)), "/", windows.n_windows)\n'
                'print("truncated windows:", int(sum(windows.truncated)), "/", windows.n_windows)\n'
                'print("source missing fraction:", windows.masked_point_fraction)\n'
                'if any(windows.truncated):\n'
                '    print("WARNING: long input series were truncated to their final 512 timestamps.")\n'
                'if any(windows.padded):\n'
                '    print("NOTE: short input series were left-padded; padding is excluded by the input mask.")'
            ),
        },
        {
            "md": (
                '## 6. Compute raw anomaly scores\n'
                '\n'
                'The core operation is `score_anomalies(..., loss="mae", channel_aggregation="none")` on the verified pinned reconstruction checkpoint (loaded in Section 3 with `task="reconstruction"`); `anomaly_score` is an uncalibrated absolute reconstruction residual per scored series/channel/timestamp. **Higher = larger reconstruction discrepancy** — higher values indicate greater anomaly evidence. The pipeline deliberately ships **no binary decision threshold** (`threshold_policy`), and positions the model could not score (pre-filled source gaps, patch-hidden neighbours, padding) are reported as unscored rather than silently dropped (`scored_point_fraction`). Successful output means this validated input was scored with the verified model under the displayed score/threshold policy; it does not mean those scores are calibrated anomaly probabilities. One more property the first table hides: a raw residual is in the **units of its channel**, and its typical size depends on how well the model reconstructs *that* channel of *that* series, so **raw residuals are comparable only within one channel of one series**. The cell therefore prints, per channel, the number of scored points and the median and maximum residual — on the default sample the clean `temperature` channel (no spike in it) has a far larger typical residual than `vibration`, the channel with the spikes, which is why Section 7 ranks within a channel.'
                '\n\n**Predict before running:** will the cell output a list of anomalies? What does a high score mean, and what does a low score *not* mean? And which channel will have the larger typical residual on the sample — the one with the spikes, or the clean one — and why?'
            ),
            "code": (
                'result = score_anomalies(windows, pipe, loss="mae", channel_aggregation="none", warmup=False)\n'
                'provenance = build_provenance(pipe, windows, result)\n'
                'scores = result.to_frame()\n'
                'print("effective model:", pipe.identity.name)\n'
                'print("effective revision:", pipe.identity.revision)\n'
                'print("verified weight file:", pipe.identity.weight_file_loaded)\n'
                'print("score policy:", result.score_policy)\n'
                'print("threshold policy:", result.threshold_policy)\n'
                'print("scored fraction:", result.scored_point_fraction)\n'
                'print(scores.head())\n'
                '\n'
                '\n'
                'def residual_summary(scores):\n'
                '    """Per-channel count, median and maximum of the scored residuals (raw residuals are comparable only within a channel)."""\n'
                '    scored_rows = scores[scores["scored"]]\n'
                '    summary = scored_rows.groupby("channel", sort=False)["anomaly_score"].agg(["count", "median", "max"])\n'
                '    summary.columns = ["scored_points", "median_residual", "max_residual"]\n'
                '    return summary\n'
                '\n'
                '\n'
                'channel_summary = residual_summary(scores)\n'
                'print("per-channel residual scale (raw residuals are comparable only within one channel of one series):")\n'
                'print(channel_summary.to_string())'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>No list of anomalies: the output is a raw residual per point and the pipeline ships no threshold. A high score means the model reconstructs that point badly — anomaly *evidence* under this score. A low score does not mean normal: smooth drift or any anomaly the model reproduces well scores low. Unscored positions (padding, pre-filled gaps, patch neighbours) are reported, not silently dropped. On the recorded Kaggle CPU run of 14 September 2026 (confirmed by the review of 2 October 2026 on local CPU) the **clean `temperature` channel** had the larger residuals — mean 1.62, maximum 3.77 — against a mean of 0.30 for `vibration`, whose three injected spikes scored 4.0 / 2.9 / 2.6: 42 clean temperature points scored above the lowest spike. Temperature has a larger amplitude (±2.5 against ±0.35) and a longer period, and the model reconstructs it poorly, so its residuals are large in absolute terms without any anomaly. That is the reason a raw residual is read only against the other residuals of the same channel.</details>'
            ),
        },
        {
            "md": (
                '## 7. Rank residuals and evaluate → evaluation report\n'
                '\n'
                "**Which channel is ranked, and why.** The ranking check is done **within one channel**, because raw residuals are comparable only there (Section 6). `SCORE_CHANNEL` chooses it; left empty, the cell takes the channel the label table names (`vibration` on the sample — the channel the spikes were injected into, which an unlabelled user would not know) and otherwise the first channel, and it prints the choice and the reason before any metric. For the bundled labelled sample, `top_k_recall` (the repository's ranking metric) ranks that channel's residuals from highest to lowest and uses `k` equal to the number of injected spikes, asking how many injected points appear among the same number of highest-scoring positions. Two contrasts are printed beside it. A **naive baseline**: the |z-score| of the raw values of the same channel (how many standard deviations each point sits from the channel mean), ranked the same way with the same k — on an isolated spike a one-line statistic can do as well as the model, and the report carries the baseline so the comparison is on record. The **pooled ranking**: all channels ranked together, which is what a user who ignores the channel rule would see — the cell prints where the injected points land in it. On BYOD every channel's highest residuals are listed (nothing is silently left out), the verdict is `not-measurable`, and the report states what labelled data or calibration would make the task measurable. `evaluation_report` is the package's public evaluation stage and always produces a report; with labels the verdict is `sample-sanity` — a falsifiable tutorial ranking check, **not** a calibrated detector metric or upstream benchmark. No arbitrary threshold is presented as universal. Successful output is a falsifiable tutorial ranking check on the labelled sample with its baseline and pooled contrast, or a ranking of every channel without a correctness claim on BYOD. The report is written to `outputs/{stem}_evaluation_report.json`."
                '\n\n**Predict before running:** three spikes were injected into `vibration`, so k = 3. How many of them do you expect among the three highest residuals of that channel — and would 3 of 3 prove the model is a good detector? Where will the spikes rank once both channels are pooled? Will the |z-score| baseline find them too?'
            ),
            "code": (
                'SCORE_CHANNEL = ""  # @param {{type:"string"}}\n'
                '# Empty = the labelled channel on the sample (vibration, where the spikes were injected), else the first channel.\n'
                '\n'
                '\n'
                'def choose_score_channel(channels, requested="", labelled="vibration"):\n'
                '    channels = list(channels)\n'
                '    if requested.strip():\n'
                '        if requested.strip() not in channels:\n'
                '            raise ValueError(f"SCORE_CHANNEL={{requested.strip()!r}} is not one of the scored channels {{channels}}.")\n'
                '        return requested.strip(), "chosen in the SCORE_CHANNEL field"\n'
                '    if labelled in channels:\n'
                '        return labelled, f"the channel the label table names ({{labelled}}: the injected spikes are in it); an unlabelled user would not know this"\n'
                '    return channels[0], "the first channel (no label table; set SCORE_CHANNEL to rank another)"\n'
                '\n'
                '\n'
                'def rank_channel(scores, channel):\n'
                '    ranked = scores[(scores["channel"] == channel) & scores["scored"]].copy().sort_values("anomaly_score", ascending=False, kind="mergesort").reset_index(drop=True)\n'
                '    ranked["rank"] = ranked.index + 1\n'
                '    return ranked\n'
                '\n'
                '\n'
                'def abs_zscore_scores(normalized, series_id, channel):\n'
                '    """A naive baseline with the same shape as the score table: |value - mean| / std of the raw channel."""\n'
                '    source = normalized[(normalized["series_id"] == series_id) & (normalized["channel"] == channel)].sort_values("timestamp")\n'
                '    values = source["value"].to_numpy(dtype=float)\n'
                '    finite = np.isfinite(values)\n'
                '    spread = float(np.nanstd(values)) or 1.0\n'
                '    z = np.abs(values - float(np.nanmean(values))) / spread\n'
                '    return pd.DataFrame({{"series_id": series_id, "timestamp": pd.to_datetime(source["timestamp"].to_numpy()), "channel": channel, "anomaly_score": z, "scored": finite}})\n'
                '\n'
                '\n'
                'channels_present = list(dict.fromkeys(scores["channel"]))\n'
                'labelled_channel = "vibration" if labels is not None else None\n'
                'score_channel, channel_reason = choose_score_channel(channels_present, SCORE_CHANNEL, labelled_channel or "")\n'
                'print("ranked channel:", score_channel, "-", channel_reason)\n'
                'print("channels not ranked for the metric (their top residuals are listed below):", [c for c in channels_present if c != score_channel])\n'
                'ranked = rank_channel(scores, score_channel)\n'
                'if ranked.empty:\n'
                '    raise ValueError("No scored positions are available after masking/padding; provide a series with observed values.")\n'
                'pooled = scores[scores["scored"]].copy().sort_values("anomaly_score", ascending=False, kind="mergesort").reset_index(drop=True)\n'
                'pooled["rank"] = pooled.index + 1\n'
                'baseline_metrics = None\n'
                'if labels is not None:\n'
                '    recall = top_k_recall(scores, labels, channel=score_channel)\n'
                '    ranked = recall["ranked"]\n'
                '    print(ranked[["rank", "timestamp", "anomaly_score", "is_injected_anomaly"]].head(12))\n'
                '    print("injected ranks within", score_channel + ":", recall["injected_ranks"])\n'
                '    print(f"tutorial top-{{recall[\'k\']}} recall ({{score_channel}}):", recall["value"])\n'
                '    marks = labels.copy()\n'
                '    marks["timestamp"] = pd.to_datetime(marks["timestamp"])\n'
                '    injected = marks[marks["is_injected_anomaly"]]\n'
                '    pooled_ranks = pooled.merge(injected, on=["series_id", "timestamp"], how="inner")\n'
                '    pooled_ranks = pooled_ranks[pooled_ranks["channel"] == score_channel]\n'
                '    print("pooled all-channel ranking: injected points rank", sorted(pooled_ranks["rank"].tolist()), "of", len(pooled), "scored positions; top-" + str(recall["k"]), "pooled recall:", float((pooled_ranks["rank"] <= recall["k"]).sum() / recall["k"]))\n'
                '    print("pooled top", recall["k"], "by channel:", pooled.head(recall["k"])["channel"].tolist())\n'
                '    baseline_recall = top_k_recall(abs_zscore_scores(normalized, windows.series_ids[0], score_channel), labels, channel=score_channel)\n'
                '    baseline_metrics = {{"top_k_recall": baseline_recall["value"]}}\n'
                '    print(f"naive |z-score| baseline on {{score_channel}} (same k = {{baseline_recall[\'k\']}}): top-k recall", baseline_recall["value"], "injected ranks", [r["rank"] for r in baseline_recall["injected_ranks"]])\n'
                '    print("VERDICT: model top-k recall", recall["value"], "vs |z-score| baseline", baseline_recall["value"], "-", "the model does not separate from a one-line statistic on this sample" if recall["value"] <= baseline_recall["value"] else "the model ranks the spikes better than the one-line statistic on this sample")\n'
                'else:\n'
                '    print(ranked[["rank", "series_id", "timestamp", "anomaly_score"]].head(12))\n'
                '    print("No labels supplied: ranking shown without a correctness metric.")\n'
                'print("highest residuals of every channel (raw residuals are not comparable across channels):")\n'
                'for channel in channels_present:\n'
                '    top = rank_channel(scores, channel).head(3)\n'
                '    print("  ", channel, [(str(row["series_id"]), str(row["timestamp"]), round(float(row["anomaly_score"]), 4)) for _, row in top.iterrows()])\n'
                '\n'
                'report = evaluation_report(result, labels, model=pipe, sample_kind=sample_kind, channel=score_channel, baseline=baseline_metrics, baseline_id="abs_zscore")\n'
                'report["ranked_channel"] = {{"channel": score_channel, "reason": channel_reason, "channels_not_ranked": [c for c in channels_present if c != score_channel]}}\n'
                'report["per_channel_residuals"] = channel_summary.to_dict("index")\n'
                'with open("outputs/{stem}_evaluation_report.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(report, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(report, indent=2, default=str))\n'
                'if report["verdict"] == "not-measurable":\n'
                '    print("No anomaly labels were supplied, so top_k_recall is not computed; the ranking above is sanity evidence only.")'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>The verdict is `sample-sanity` whatever the count: the spikes are large, isolated jumps added to a smooth synthetic series, which is the easiest case a reconstruction model can face. On the recorded runs (Kaggle CPU 14 September 2026; the review of 2 October 2026) the three spikes were the top 3 of `vibration` (recall 1.0) — but a plain |z-score| on the same channel also got all three, so recall 1.0 does not separate MOMENT from a one-line statistic here, and in the **pooled** ranking the spikes sat at 1st, 21st and 45th, behind 42 clean temperature points. Even 3 of 3 is one falsifiable check on one series, not a calibrated detector metric; your own data needs labelled true anomalies *and* hard normal events.</details>\n\n**Change one thing (optional):** the channel aggregation. Re-run Section 6 with `channel_aggregation="mean"` and then `"max"` in `score_anomalies(...)`, then Sections 7–8 (with `SCORE_CHANNEL` left empty the ranked "channel" becomes the aggregate). Predict first: with the clean channel\'s large residuals mixed in, will all three spikes still rank in the top 3? In the review of 2 October 2026 the recall fell to 0.667 under `mean` and 0.333 under `max`. (Switching `loss="mse"` instead cannot change the ranking while channels are *not* aggregated: under `channel_aggregation="none"` the MSE residual is the square of the MAE residual, a monotone transform — it rescales the scores and leaves the order.)'
            ),
        },
        {
            "md": (
                '## 8. Visualize raw score over time\n'
                '\n'
                'One plot per series (up to four) of the ranked channel: the raw residual against time, with the raw value of the channel drawn beneath it, the score axis labelled, the first and last timestamps on the x-axis and, on the labelled sample, the injected timestamps marked. Series are never joined into one line. The plot adds no decision threshold. Smooth drift or other real anomalies that the reconstruction model reproduces well may score low, while benign but hard-to-reconstruct events may score high. Successful rendering makes score behaviour easier to inspect; it does not replace the machine-readable score table or establish detector calibration.'
            ),
            "code": (
                'def write_score_svg(path, stamps, score, *, title, value=None, marks=(), width=760, height=320):\n'
                '    """Raw score over time (upper panel, labelled axis) with the raw channel value beneath (lower panel); `marks` are timestamps to circle."""\n'
                '    stamps = list(stamps)\n'
                '    n = len(stamps)\n'
                '    left, right = 64, width - 20\n'
                '    panels = [("raw residual score", score, 30, height // 2 - 10, "#dc2626")] + ([("raw value", value, height // 2 + 20, height - 36, "#111827")] if value is not None else [])\n'
                '    svg = [f\'<svg xmlns="http://www.w3.org/2000/svg" width="{{width}}" height="{{height}}">\', f\'<text x="{{left}}" y="18" font-family="sans-serif" font-size="14">{{title}}</text>\']\n'
                '\n'
                '    def x_of(index):\n'
                '        return left + (right - left) * index / max(n - 1, 1)\n'
                '\n'
                '    for label, values, top, bottom, stroke in panels:\n'
                '        finite = [float(v) for v in values if v is not None and np.isfinite(v)]\n'
                '        low, high = (min(finite), max(finite)) if finite else (0.0, 1.0)\n'
                '        span = high - low or 1.0\n'
                '\n'
                '        def y_of(v, low=low, span=span, top=top, bottom=bottom):\n'
                '            return bottom - (bottom - top) * (float(v) - low) / span\n'
                '\n'
                '        svg.append(f\'<line x1="{{left}}" y1="{{top}}" x2="{{left}}" y2="{{bottom}}" stroke="#9ca3af"/>\')\n'
                '        for tick in (low, (low + high) / 2, high):\n'
                '            svg.append(f\'<text x="4" y="{{y_of(tick) + 4:.1f}}" font-family="sans-serif" font-size="10">{{tick:.3g}}</text>\')\n'
                '        svg.append(f\'<text x="{{left + 6}}" y="{{top + 12}}" font-family="sans-serif" font-size="11" fill="{{stroke}}">{{label}}</text>\')\n'
                '        segment = []\n'
                '        for i, v in enumerate(values):\n'
                '            if v is None or not np.isfinite(v):\n'
                '                if len(segment) > 1:\n'
                '                    svg.append(f\'<polyline fill="none" stroke="{{stroke}}" stroke-width="1.5" points="{{" ".join(segment)}}"/>\')\n'
                '                segment = []\n'
                '                continue\n'
                '            segment.append(f"{{x_of(i):.1f}},{{y_of(v):.1f}}")\n'
                '        if len(segment) > 1:\n'
                '            svg.append(f\'<polyline fill="none" stroke="{{stroke}}" stroke-width="1.5" points="{{" ".join(segment)}}"/>\')\n'
                '        for i, stamp in enumerate(stamps):\n'
                '            if stamp in marks and values[i] is not None and np.isfinite(values[i]):\n'
                '                svg.append(f\'<circle cx="{{x_of(i):.1f}}" cy="{{y_of(values[i]):.1f}}" r="5" fill="none" stroke="#2563eb" stroke-width="2"/>\')\n'
                '    svg.append(f\'<text x="{{left}}" y="{{height - 8}}" font-family="sans-serif" font-size="10">{{stamps[0] if stamps else ""}}</text>\')\n'
                '    svg.append(f\'<text x="{{right - 150}}" y="{{height - 8}}" font-family="sans-serif" font-size="10">{{stamps[-1] if stamps else ""}}</text>\')\n'
                '    if marks:\n'
                '        svg.append(f\'<text x="{{right - 150}}" y="18" font-family="sans-serif" font-size="11" fill="#2563eb">o = labelled spike</text>\')\n'
                '    svg.append("</svg>")\n'
                '    Path(path).parent.mkdir(parents=True, exist_ok=True)\n'
                '    Path(path).write_text("\\n".join(svg), encoding="utf-8")\n'
                '    return Path(path)\n'
                '\n'
                '\n'
                'marked = set()\n'
                'if labels is not None:\n'
                '    marked = set(pd.to_datetime(labels.loc[labels["is_injected_anomaly"], "timestamp"]).tolist())\n'
                'score_plots = []\n'
                'series_shown = list(dict.fromkeys(ranked["series_id"]))[:4]\n'
                'for series_id in series_shown:\n'
                '    ordered = ranked[ranked["series_id"] == series_id].sort_values("timestamp")\n'
                '    raw = normalized[(normalized["series_id"] == series_id) & (normalized["channel"] == score_channel)].set_index("timestamp")["value"] if score_channel in set(normalized["channel"]) else None\n'
                '    raw_values = [float(raw.get(stamp, np.nan)) for stamp in ordered["timestamp"]] if raw is not None else None\n'
                '    score_plots.append(write_score_svg(f"outputs/moment_anomaly_scores_{{series_id}}.svg", ordered["timestamp"].tolist(), ordered["anomaly_score"].tolist(), title=f"MOMENT raw anomaly score - series {{series_id}}, channel {{score_channel}}", value=raw_values, marks=marked))\n'
                'score_plot = score_plots[0]\n'
                'if len(set(ranked["series_id"])) > len(series_shown):\n'
                '    print("series not plotted:", sorted(set(ranked["series_id"]) - set(series_shown)))\n'
                'try:\n'
                '    from IPython.display import SVG, display\n'
                '\n'
                '    for path in score_plots:\n'
                '        display(SVG(filename=str(path)))\n'
                'except ImportError:\n'
                '    print("SVGs written:", [str(path) for path in score_plots])'
            ),
        },
        {
            "md": (
                '## 9. Export outputs and provenance\n'
                '\n'
                "This stage writes `outputs/moment_anomaly_scores.csv` (identifier-preserving residual scores with the `scored` flag) and `outputs/moment_anomaly_provenance.json` (plus the per-series diagnostic SVGs), and `outputs/{stem}_result.json` with the input manifest, the evaluation report, the sample identity and digests, the notebook's source (repository, revision, embedded module digests, generator), the model identifier, the immutable model revision and licence, and the runtime identity. Export success does **not** establish that residuals are calibrated anomaly decisions or that the bundled ranking result generalizes. No credentials are recorded."
            ),
            "code": (
                'scores.to_csv("outputs/moment_anomaly_scores.csv", index=False)\n'
                'provenance["data"] = sample_identity\n'
                'provenance["evaluation"] = {{"estimation_procedure": "deterministic injected-spike ranking on the synthetic sample" if labels is not None else "unlabelled BYOD residual ranking; no correctness metric", "sample_evidence_only": True, "top_k_recall": None if labels is None else report["metrics"][0]["value"]}}\n'
                'provenance["notebook_source"] = NOTEBOOK_SOURCE\n'
                'with open("outputs/moment_anomaly_provenance.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(provenance, handle, indent=2, default=str)\n'
                'payload_out = {{\n'
                '    "result": {{"n_windows": len(result.window_ids), "score_channel": score_channel, "score_channel_reason": channel_reason, "per_channel_residuals": channel_summary.to_dict("index"), "zscore_baseline": baseline_metrics, "loss": result.loss, "channel_aggregation": result.channel_aggregation, "scored_point_fraction": result.scored_point_fraction, "threshold_policy": result.threshold_policy, "top_ranked": ranked[["rank", "series_id", "timestamp", "anomaly_score"]].head(12).to_dict("records")}},\n'
                '    "evaluation_report": report,\n'
                '    "input_manifest": input_manifest,\n'
                '    "sample": {{"kind": sample_kind, **sample_identity}},\n'
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                '    "model_id": MODEL_ID,\n'
                '    "model_revision": MODEL_REVISION,\n'
                '    "model_license": MODEL_LICENSE,\n'
                '    "runtime": {{"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "pandas": pandas.__version__, "numpy": numpy.__version__, "device": pipe.identity.device, "dtype": pipe.identity.dtype}},\n'
                '}}\n'
                'with open("outputs/moment_anomaly_detection_result.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(payload_out, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(sorted(os.listdir("outputs")))'
            ),
        },
    ],
    "closing": (
        '## Interpretation and limits\n'
        '\n'
        "A successful run proves that the carried package can validate this input, resolve and integrity-check the pinned MOMENT reconstruction checkpoint, compute the documented raw residual score over its valid scored domain, rank those scores, and export identifier-preserving outputs with provenance; on the bundled sample it also provides a falsifiable check of how the three injected spikes rank. Successful execution proves that the recorded repository revision's package, carried in this notebook, can do exactly that — without the repository being reachable — and no more.\n"
        '\n'
        'It **does not prove** that high residuals are real-world anomalies, that low residuals are normal, that the bundled top-k result generalizes, or that any numerical threshold is calibrated; the evaluation report says `sample-sanity` on the labelled sample and `not-measurable` without labels for that reason. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain. Deployment thresholds, if needed, belong to the downstream application and require representative calibration/validation data and an explicit false-positive/false-negative cost model.\n'
        '\n'
        '**Next experiments:** re-run Section 6 with `channel_aggregation="mean"` and then `"max"`, then Sections 7–8, and compare the recall and the ranks with the per-channel run (predict first; see the activity after Section 7); set `SCORE_CHANNEL = "temperature"` and read what the top-3 of a channel with no anomaly in it looks like; build a domain-specific labelled validation set containing both true anomalies and difficult normal events, then evaluate ranking and calibration separately; enable `USE_BYOD` with an unlabelled series and read the `not-measurable` report and every channel\'s top residuals. (`loss="mse"` only rescales the per-channel scores while channels are not aggregated, so it changes no ranking there.)\n'
        '\n'
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **Section 1 fails while building `momentfm`** — the one source dependency is built from its pinned upstream commit and needs `git` and access to its Git host; run Section 1 again, or use Colab or Kaggle (both ship `git`).\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        '- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n'
        '- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, copy the CSV into the runtime and set `BYOD_PATH`.\n'
        '- **BYOD: "Upload exactly one CSV"** — the dialog was cancelled or several files were chosen; run Section 4 again.\n'
        '- **BYOD: a `ValidationError`** — it carries a code and names the column, series or rule (duplicate header, duplicate row, a ceiling, an all-missing channel); fix the CSV.\n\n'
        '## Glossary\n\n'
        '- **Long-format table** — one row per `(series_id, timestamp, channel, value)`; the package turns it into windows.\n'
        '- **Window / patch** — the 512-step input MOMENT reads per series, cut into 64 non-overlapping 8-step patches.\n'
        '- **Left padding / truncation** — a series shorter than 512 steps is padded at the start and the padding masked out; a longer one keeps its final 512 timestamps.\n'
        '- **Instance normalisation** — each window is rescaled by its own mean and spread before the encoder, so the absolute level is removed.\n'
        '- **Input manifest** — the record of what validation saw and changed (padding, truncation, missingness) before the model ran.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        "- **Reconstruction residual** — the absolute difference between a value and the model's reconstruction of it; the raw anomaly score here, in the channel's units and comparable only within one channel of one series.\n"
        '- **Channel aggregation** — `none` keeps one score per channel; `mean` / `max` collapse the channels to one score per timestamp, which lets a noisy channel drown a spike in another.\n'
        '- **|z-score| baseline** — distance of a raw value from its channel mean in standard deviations, ranked the same way; the one-line statistic the model is compared with.\n'
        '- **Threshold** — a cut-off that turns scores into anomaly labels; none ships, because it needs representative calibration data and a cost model.\n'
        '- **Top-k recall** — with k equal to the number of injected spikes, the share of them among the k highest scores.\n'
        '- **`sample-sanity` / `not-measurable`** — the verdicts with labels (one falsifiable check) and without them.\n'
        '- **BYOD** — bring your own data: your long-format CSV through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- Of the ___ injected spikes, ___ ranked among the top ___ scores of `vibration` (top-k recall ___); the |z-score| baseline found ___; in the pooled ranking they sat at ranks ___.\n'
        '- The channel with the larger typical residual was ___ (median ___ vs ___), although it holds no anomaly; I think that is because ___.\n'
        '- The highest-scoring point that was not a spike was at ___; I think it scored high because ___.\n'
        '- Before using a threshold on my own data I would need ___.\n'
        '\n'
        '## References\n'
        '\n'
        '- Repository README: https://github.com/kurtvalcorza/moment-pipeline/blob/main/README.md\n'
        '- Repository model card: https://github.com/kurtvalcorza/moment-pipeline/blob/main/MODEL_CARD.md\n'
        '- Sample dataset card: https://github.com/kurtvalcorza/moment-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n'
        '- Upstream model: https://huggingface.co/{MODEL_ID}\n'
        '- Upstream library: https://github.com/moment-timeseries-foundation-model/moment\n'
        '- MOMENT: A Family of Open Time-series Foundation Models: https://arxiv.org/abs/2402.03885'
    ),
}

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
        '**Who this notebook is for.** A learner who knows basic Python and pandas, has used Colab or Jupyter and has met time-series data, and wants to see how a pretrained time-series foundation model is used without any training to rank unusual points — what goes in, what comes out, and what the output does and does not prove. The audience is students and practitioners preparing their own sensor or monitoring series; no prior experience with MOMENT is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | a long-format CSV: the synthetic series `A` with three injected `vibration` spikes (and their labels) or your own unlabelled CSV |\n| Model | the MOMENT-1-base reconstruction head (`task="reconstruction"`); each point is scored by how badly the model reconstructs it |\n| Output | a raw, uncalibrated residual score per series, channel and timestamp (higher = stronger anomaly evidence), a top-k recall check on the labelled sample, and no threshold |\n\n**How to use this notebook.** Choose a runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook\'s own Python, so no restart is needed (the recorded hosted runs of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the default path. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer. No per-cell outputs of the hosted runs are recorded in this repository (only their pass/fail), so the worked answers state what the code and the sample guarantee rather than quoting numbers; compare them with what your run prints. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the synthetic sample with injected spikes (or your CSV) → 5 validation and canonicalization *(core concept: windows, padding, masks)* → 6 raw residual scores *(core concept: a score is not a decision)* → 7 ranking and top-k recall *(evaluation practice)* → 8 the plot → 9 → export and provenance *(engineering)* → conclude.'
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
        '- **Compute:** CPU is the default path and no GPU is required; CUDA is used automatically when available. The public v1 API accepts `float32` only. The pinned `torch==2.14.0` install is the largest download of the run, followed by the ~454 MB `model.safetensors`.',
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
                'The core operation is `score_anomalies(..., loss="mae", channel_aggregation="none")` on the verified pinned reconstruction checkpoint (loaded in Section 3 with `task="reconstruction"`); `anomaly_score` is an uncalibrated absolute reconstruction residual per scored series/channel/timestamp. **Higher = larger reconstruction discrepancy** — higher values indicate greater anomaly evidence. The pipeline deliberately ships **no binary decision threshold** (`threshold_policy`), and positions the model could not score (pre-filled source gaps, patch-hidden neighbours, padding) are reported as unscored rather than silently dropped (`scored_point_fraction`). Successful output means this validated input was scored with the verified model under the displayed score/threshold policy; it does not mean those scores are calibrated anomaly probabilities.'
                '\n\n**Predict before running:** will the cell output a list of anomalies? What does a high score mean, and what does a low score *not* mean?'
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
                'print(scores.head())'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>No list of anomalies: the output is a raw residual per point and the pipeline ships no threshold. A high score means the model reconstructs that point badly — anomaly *evidence* under this score. A low score does not mean normal: smooth drift or any anomaly the model reproduces well scores low. Unscored positions (padding, pre-filled gaps, patch neighbours) are reported, not silently dropped.</details>'
            ),
        },
        {
            "md": (
                '## 7. Rank residuals and evaluate → evaluation report\n'
                '\n'
                "For the bundled labelled sample, `top_k_recall` (the repository's ranking metric) ranks `vibration` residuals from highest to lowest and uses `k` equal to the number of injected spikes, asking how many injected points appear among the same number of highest-scoring positions. `evaluation_report` is the package's public evaluation stage and always produces a report: with labels it carries `top_k_recall` with the verdict `sample-sanity` — a falsifiable tutorial ranking check, **not** a calibrated detector metric or upstream benchmark; for BYOD without labels only the highest residuals are displayed, the verdict is `not-measurable`, and the report states what labelled data or calibration would make the task measurable. No arbitrary threshold is presented as universal. Successful output is a falsifiable tutorial ranking check on the labelled sample, or a ranking without a correctness claim on BYOD. The report is written to `outputs/{stem}_evaluation_report.json`."
                '\n\n**Predict before running:** three spikes were injected into `vibration`, so k = 3. How many of them do you expect among the three highest residuals — and would 3 of 3 prove the model is a good detector?'
            ),
            "code": (
                'score_channel = "vibration" if "vibration" in set(scores["channel"]) else str(scores["channel"].iloc[0])\n'
                'ranked = scores[(scores["channel"] == score_channel) & scores["scored"]].copy().sort_values("anomaly_score", ascending=False).reset_index(drop=True)\n'
                'if ranked.empty:\n'
                '    raise ValueError("No scored positions are available after masking/padding; provide a series with observed values.")\n'
                'ranked["rank"] = ranked.index + 1\n'
                'if labels is not None:\n'
                '    recall = top_k_recall(scores, labels, channel=score_channel)\n'
                '    ranked = recall["ranked"]\n'
                '    print(ranked[["rank", "timestamp", "anomaly_score", "is_injected_anomaly"]].head(12))\n'
                '    print("injected ranks:", recall["injected_ranks"])\n'
                '    print(f"tutorial top-{{recall[\'k\']}} recall:", recall["value"])\n'
                'else:\n'
                '    print(ranked[["rank", "series_id", "timestamp", "anomaly_score"]].head(12))\n'
                '    print("No labels supplied: ranking shown without a correctness metric.")\n'
                '\n'
                'report = evaluation_report(result, labels, model=pipe, sample_kind=sample_kind, channel=score_channel)\n'
                'with open("outputs/{stem}_evaluation_report.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(report, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(report, indent=2, default=str))\n'
                'if report["verdict"] == "not-measurable":\n'
                '    print("No anomaly labels were supplied, so top_k_recall is not computed; the ranking above is sanity evidence only.")'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>The verdict is `sample-sanity` whatever the count: the spikes are large, isolated jumps added to a smooth synthetic series, which is the easiest case a reconstruction model can face. Even 3 of 3 is one falsifiable check on one series, not a calibrated detector metric; your own data needs labelled true anomalies *and* hard normal events.</details>'
            ),
        },
        {
            "md": (
                '## 8. Visualize raw score over time\n'
                '\n'
                'The diagnostic plot shows the raw residual and adds no decision threshold. Smooth drift or other real anomalies that the reconstruction model reproduces well may score low, while benign but hard-to-reconstruct events may score high. Successful rendering makes score behaviour easier to inspect; it does not replace the machine-readable score table or establish detector calibration.'
            ),
            "code": (
                'def write_line_svg(path, layers, *, title, width=760, height=280):\n'
                '    all_values = [float(value) for _, values in layers for value in values]\n'
                '    low, high = min(all_values), max(all_values)\n'
                '    span = high - low or 1.0\n'
                '    max_points = max(len(values) for _, values in layers)\n'
                '    left, right, top, bottom = 48, width - 20, 30, height - 38\n'
                '\n'
                '    def point(index, value):\n'
                '        x = left + (right - left) * index / max(max_points - 1, 1)\n'
                '        y = bottom - (bottom - top) * (float(value) - low) / span\n'
                '        return f"{{x:.1f}},{{y:.1f}}"\n'
                '\n'
                '    svg = [f\'<svg xmlns="http://www.w3.org/2000/svg" width="{{width}}" height="{{height}}">\', f\'<text x="{{left}}" y="18" font-family="sans-serif" font-size="14">{{title}}</text>\']\n'
                '    for idx, (label, values) in enumerate(layers):\n'
                '        points = " ".join(point(i, value) for i, value in enumerate(values))\n'
                '        stroke = ["#111827", "#2563eb", "#dc2626"][idx % 3]\n'
                '        svg.append(f\'<polyline fill="none" stroke="{{stroke}}" stroke-width="2" points="{{points}}"/>\')\n'
                '        svg.append(f\'<text x="{{left + 180 * idx}}" y="{{height - 10}}" font-family="sans-serif" font-size="12" fill="{{stroke}}">{{label}}</text>\')\n'
                '    svg.append("</svg>")\n'
                '    Path(path).parent.mkdir(parents=True, exist_ok=True)\n'
                '    Path(path).write_text("\\n".join(svg), encoding="utf-8")\n'
                '    return Path(path)\n'
                '\n'
                '\n'
                'ordered = ranked.sort_values("timestamp")\n'
                'score_plot = write_line_svg("outputs/moment_anomaly_scores.svg", [("raw MAE residual", ordered["anomaly_score"].fillna(0.0).tolist())], title=f"MOMENT raw anomaly score - {{score_channel}}")\n'
                'try:\n'
                '    from IPython.display import SVG, display\n'
                '\n'
                '    display(SVG(filename=str(score_plot)))\n'
                'except ImportError:\n'
                '    print(f"SVG written to {{score_plot}}")'
            ),
        },
        {
            "md": (
                '## 9. Export outputs and provenance\n'
                '\n'
                "This stage writes `outputs/moment_anomaly_scores.csv` (identifier-preserving residual scores with the `scored` flag) and `outputs/moment_anomaly_provenance.json` (plus the diagnostic SVG), and `outputs/{stem}_result.json` with the input manifest, the evaluation report, the sample identity and digests, the notebook's source (repository, revision, embedded module digests, generator), the model identifier, the immutable model revision and licence, and the runtime identity. Export success does **not** establish that residuals are calibrated anomaly decisions or that the bundled ranking result generalizes. No credentials are recorded."
            ),
            "code": (
                'scores.to_csv("outputs/moment_anomaly_scores.csv", index=False)\n'
                'provenance["data"] = sample_identity\n'
                'provenance["evaluation"] = {{"estimation_procedure": "deterministic injected-spike ranking on the synthetic sample" if labels is not None else "unlabelled BYOD residual ranking; no correctness metric", "sample_evidence_only": True, "top_k_recall": None if labels is None else report["metrics"][0]["value"]}}\n'
                'provenance["notebook_source"] = NOTEBOOK_SOURCE\n'
                'with open("outputs/moment_anomaly_provenance.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(provenance, handle, indent=2, default=str)\n'
                'payload_out = {{\n'
                '    "result": {{"n_windows": len(result.window_ids), "score_channel": score_channel, "loss": result.loss, "channel_aggregation": result.channel_aggregation, "scored_point_fraction": result.scored_point_fraction, "threshold_policy": result.threshold_policy, "top_ranked": ranked[["rank", "series_id", "timestamp", "anomaly_score"]].head(12).to_dict("records")}},\n'
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
        '**Next experiments:** build a domain-specific labelled validation set containing both true anomalies and difficult normal events, then evaluate ranking and calibration separately; enable `USE_BYOD` with an unlabelled series and read the `not-measurable` report; switch `loss="mse"` and compare how the injected spikes rank.\n'
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
        "- **Reconstruction residual** — the absolute difference between a value and the model's reconstruction of it; the raw anomaly score here.\n"
        '- **Threshold** — a cut-off that turns scores into anomaly labels; none ships, because it needs representative calibration data and a cost model.\n'
        '- **Top-k recall** — with k equal to the number of injected spikes, the share of them among the k highest scores.\n'
        '- **`sample-sanity` / `not-measurable`** — the verdicts with labels (one falsifiable check) and without them.\n'
        '- **BYOD** — bring your own data: your long-format CSV through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- Of the ___ injected spikes, ___ ranked among the top ___ scores (top-k recall ___).\n'
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

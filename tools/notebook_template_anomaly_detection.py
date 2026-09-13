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
    'learning_objectives': (
        'install the pinned runtime; read what the carried package guarantees; resolve and digest-verify the immutable model revision; generate the deterministic labelled synthetic sample or bring your own long-format CSV; validate and canonicalize the input into an input manifest; compute raw anomaly scores through the production-facing API; interpret score direction and threshold semantics; check the injected-spike ranking quantitatively through an evaluation report that is `sample-sanity` only when labels exist; and export raw scores with provenance. By the end of this notebook you will be able to do each of these without the repository being reachable.'
    ),
    'exclusions': (
        'a calibrated detector, a universal threshold, forecasting, classification, or production fitness. High reconstruction error and real-world anomaly status are not equivalent concepts.'
    ),
    'prerequisites': [
        '- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is the default path and no GPU is required; CUDA is used automatically when available. The public v1 API accepts `float32` only. The pinned `torch==2.14.0` install is the largest download of the run, followed by the ~454 MB `model.safetensors`.',
        '- **Knowledge:** basic Python and pandas; what a long-format time-series table is.',
        "- **Data:** the default sample is the repository's deterministic two-channel synthetic series with three documented injected spikes in the `vibration` channel, regenerated in code together with its label table, so nothing is downloaded and no private data is needed. Labels make ranking behaviour falsifiable; they are not calibration data or benchmark evidence. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 CSV with columns `series_id`, `timestamp`, `channel`, `value`; `value` numeric or missing; identifiers and timestamps valid. Duplicate or ambiguous column names are rejected from the raw CSV header before dataframe parsing, and duplicate `(series_id, channel, timestamp)` rows are rejected by production validation. BYOD does not require anomaly labels; without labels the notebook ranks residuals but cannot measure detector quality. Operational ceilings: at most 5,000,000 rows, 1,024 series, 32 channels, and 1,024 canonical windows; MOMENT uses 512-step windows and 8-step non-overlapping patches; short series are left-padded, long series keep the final 512 timestamps, irregular spacing is surfaced rather than silently interpolated. BYOD is read locally in the notebook runtime and is not sent to an external inference service. Do not upload confidential or restricted data (personal or otherwise sensitive data included) to a hosted notebook environment unless you are authorized to do so.",
    ],
    "cells": [
        {
            "md": (
                '## 4. Generate the synthetic sample or optional BYOD\n'
                '\n'
                "The default sample is **synthetic**: the repository's `examples/sample-data/generate_samples.py` formulas (one series `A`, channels `vibration` and `temperature`, 256 steps at 15 minutes — a trend plus two sinusoids) regenerated in code and rendered to the same canonical CSV bytes, so the `moment_anomaly.csv` SHA-256 is asserted against the digest the repository checks in. The three injected spikes in the `vibration` channel come with a label table, digest-asserted the same way, so the ranking check later is falsifiable; BYOD carries no labels. It is deterministic teaching data, not benchmark evidence. The optional upload path first validates the **raw CSV header** with `read_long_csv_bytes()` (duplicate or ambiguous names cannot be silently renamed by pandas); the resulting frame still goes through the same production validation/canonicalization path in the next stage. Set `USE_BYOD=True` in Colab, or set `DIMER_BYOD_PATH` in automation. Successful completion means one identified input frame is available and its source/digest are recorded; look for the sample identity (kind, name, digest) and the first rows."
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
                'BYOD_PATH = os.environ.get("DIMER_BYOD_PATH")\n'
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
                '    payload = Path(BYOD_PATH).read_bytes()\n'
                '    frame = read_long_csv_bytes(payload)\n'
                '    labels = None  # BYOD carries no anomaly labels; the ranking is shown without a correctness metric\n'
                '    sample_identity = {{"kind": "byod", "name": Path(BYOD_PATH).name, "sha256": hashlib.sha256(payload).hexdigest()}}\n'
                '    sample_kind = "BYOD"\n'
                'elif USE_BYOD:\n'
                '    from google.colab import files\n'
                '    uploaded = files.upload()\n'
                '    if len(uploaded) != 1:\n'
                '        raise ValueError("Upload exactly one CSV with columns series_id,timestamp,channel,value.")\n'
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
                '## 7. Rank residuals and evaluate → evaluation report\n'
                '\n'
                "For the bundled labelled sample, `top_k_recall` (the repository's ranking metric) ranks `vibration` residuals from highest to lowest and uses `k` equal to the number of injected spikes, asking how many injected points appear among the same number of highest-scoring positions. `evaluation_report` is the package's public evaluation stage and always produces a report: with labels it carries `top_k_recall` with the verdict `sample-sanity` — a falsifiable tutorial ranking check, **not** a calibrated detector metric or upstream benchmark; for BYOD without labels only the highest residuals are displayed, the verdict is `not-measurable`, and the report states what labelled data or calibration would make the task measurable. No arbitrary threshold is presented as universal. Successful output is a falsifiable tutorial ranking check on the labelled sample, or a ranking without a correctness claim on BYOD. The report is written to `outputs/{stem}_evaluation_report.json`."
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

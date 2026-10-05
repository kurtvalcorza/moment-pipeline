"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

This is the embeddings tutorial of moment-pipeline; the repository ships three TASK-INFERENCE notebooks
(embeddings, imputation, anomaly scoring) generated from three templates that share the same carried
package (ten modules under src/moment_pipeline/, in dependency order) and the same model cell.

Generator /2 keys in use: ``modules`` lists every module except ``__init__.py``; ``entry_module`` is
``model.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/``MODEL_LICENSE``/``MODEL_KEY``); ``model_load`` is
the package's documented loader ``load_moment(task="embedding", weights_dir=WEIGHTS_DIR)`` — MOMENT
replaces its head per task (RFC M-8), so the task is part of the loaded model's identity.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    'package': 'moment_pipeline',
    'repo_name': 'moment-pipeline',
    'stem': 'moment_embeddings',
    'notebook_name': 'moment_embeddings_colab.ipynb',
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
    'model_load': 'load_moment(task="embedding", weights_dir=WEIGHTS_DIR)',
    'runtime_imports': ['torch', 'transformers', 'pandas', 'numpy'],
    'title': 'MOMENT embeddings — DIMER task-inference tutorial (standalone)',
    'badges': [
        (
            'GitHub',
            'https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/kurtvalcorza/moment-pipeline',
        ),
        (
            'Open In Colab',
            'https://colab.research.google.com/assets/colab-badge.svg',
            'https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_embeddings_colab.ipynb',
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
        'pretrained time-series representation extraction (pooled window embeddings) with the pinned `AutonLab/MOMENT-1-base` encoder'
    ),
    'intro': (
        "This notebook extracts pooled representations from the pretrained MOMENT encoder through the repository's public `moment_pipeline` API, carried in this notebook. **No gradient training, fine-tuning, in-context conditioning, or fitted preprocessing occurs** — **no adaptation occurs:** input validation/canonicalization is deterministic preprocessing only. **Upstream vs. this repository.** Upstream MOMENT supplies the pretrained encoder and embedding operation. This repository supplies the immutable model pin, safetensors integrity checks, long-format input validation/canonicalization, missingness disclosures, output schema, and provenance export. Embeddings are representations, not predictions: there is no intrinsic accuracy metric, and the evaluation report says so."
    ),
    "guided": {"opening": [(
        '**Who this notebook is for.** A learner who knows basic Python and pandas, has used Colab or Jupyter and has met time-series data, and wants to see how a pretrained time-series foundation model is used without any training — what goes in, what comes out, and what the output does and does not prove. The audience is students and practitioners preparing their own sensor or monitoring series; no prior experience with MOMENT is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | a long-format CSV (`series_id, timestamp, channel, value`): the synthetic two-channel series `A` (256 steps at 15 minutes) or your own |\n| Model | the MOMENT-1-base encoder loaded with `task="embedding"` (its head is an identity); each channel is embedded independently |\n| Output | one mean-pooled vector per 512-step window, written with its `series_id` and `window_id`, and an evaluation report that is always `not-measurable` |\n\n**How to use this notebook.** Choose a runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook\'s own Python, so no restart is needed (the recorded hosted runs of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the default path. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer. No per-cell outputs of the hosted runs are recorded in this repository (only their pass/fail), so the worked answers state what the code and the sample guarantee rather than quoting numbers; compare them with what your run prints. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the synthetic sample (or your CSV) → 5 validation and canonicalization with a deliberate refusal *(core concept: windows, padding, masks)* → 6 embeddings *(core concept: representation, not prediction)* → 7 the `not-measurable` report *(evaluation practice)* → 8 → export and provenance *(engineering)* → conclude.'
    )]},
    'learning_objectives': (
        'install the pinned runtime; read what the carried package guarantees; resolve and digest-verify the immutable model revision; generate the deterministic synthetic sample or bring your own long-format CSV; validate and canonicalize the input into an input manifest; extract pooled MOMENT embeddings through the production-facing API; interpret embedding shape, pooling, channel, and missingness semantics; produce an evaluation report that is always `not-measurable` for representations; and export identifier-preserving embeddings and provenance. By the end of this notebook you will be able to do each of these without the repository being reachable.'
    ),
    'exclusions': (
        'classification, forecasting, anomaly decisions, fine-tuning, or evidence that the embeddings are suitable for any specific downstream task. Embeddings are representations, not predictions.'
    ),
    'prerequisites': [
        '- **Learner:** basic Python and pandas familiarity; no prior experience with MOMENT or time-series foundation models. Windows, patches, padding, masks and the evaluation verdicts are explained where they are first used and again in the Glossary.',
        '- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels (plus `momentfm`, built from its pinned commit), so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported.',
        '- **Compute:** CPU is the default path and no GPU is required; CUDA is used automatically when available. The public v1 API accepts `float32` only. The pinned `torch==2.14.0` install is the largest download of the run, followed by the ~454 MB `model.safetensors`.',
        '- **Knowledge:** basic Python and pandas; what a long-format time-series table is.',
        "- **Data:** the default sample is the repository's deterministic two-channel synthetic series regenerated in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 CSV with columns `series_id`, `timestamp`, `channel`, `value`; `value` numeric or missing; identifiers and timestamps valid. Duplicate or ambiguous column names are rejected from the raw CSV header before dataframe parsing, and duplicate `(series_id, channel, timestamp)` rows are rejected by production validation. Operational ceilings: at most 5,000,000 rows, 1,024 series, 32 channels, and 1,024 canonical windows; MOMENT uses 512-step windows and 8-step non-overlapping patches; short series are left-padded, long series keep the final 512 timestamps, irregular spacing is surfaced rather than silently interpolated. BYOD is read locally in the notebook runtime and is not sent to an external inference service. Do not upload confidential or restricted data (personal or otherwise sensitive data included) to a hosted notebook environment unless you are authorized to do so.",
    ],
    "cells": [
        {
            "md": (
                '## 4. Generate the synthetic sample or optional BYOD\n'
                '\n'
                "The default sample is **synthetic**: the repository's `examples/sample-data/generate_samples.py` formulas (one series `A`, channels `vibration` and `temperature`, 256 steps at 15 minutes — a trend plus two sinusoids) regenerated in code and rendered to the same canonical CSV bytes, so the `moment_clean.csv` SHA-256 is asserted against the digest the repository checks in. It is deterministic teaching data, not benchmark evidence. The optional upload path first validates the **raw CSV header** with `read_long_csv_bytes()` (duplicate or ambiguous names cannot be silently renamed by pandas); the resulting frame still goes through the same production validation/canonicalization path in the next stage. Set `USE_BYOD=True` and `BYOD_PATH` to a CSV already in the runtime (Colab, Kaggle or Jupyter); on Colab an empty `BYOD_PATH` opens an upload dialog, and a cancelled upload stops with a message. `DIMER_BYOD_PATH` remains the automation hook. Successful completion means one identified input frame is available and its source/digest are recorded; look for the sample identity (kind, name, digest) and the first rows."
                '\n\n**Predict before running:** the sample has 256 timestamps per channel and MOMENT reads 512-step windows. How many windows will series `A` give, and what has to happen to the missing half?'
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
                'SAMPLE_SHA256 = "34fc475828d2f62108b340efd255501a898577be11286c034da2e0f766ee963c"  # examples/sample-data/SHA256SUMS\n'
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
                '    labels = None\n'
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
                '    labels = None\n'
                '    sample_identity = {{"kind": "byod", "name": name, "sha256": hashlib.sha256(payload).hexdigest()}}\n'
                '    sample_kind = "BYOD"\n'
                'else:\n'
                '    samples = build_samples()\n'
                '    payload = sample_csv_bytes(samples["moment_clean.csv"])\n'
                '    observed = hashlib.sha256(payload).hexdigest()\n'
                '    if observed != SAMPLE_SHA256:\n'
                '        raise ValueError(f"Synthetic sample digest mismatch: {{observed}} != {{SAMPLE_SHA256}}")\n'
                '    frame = read_long_csv_bytes(payload)\n'
                '    sample_identity = {{"kind": "synthetic", "name": "moment_clean.csv", "sha256": observed}}\n'
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
                "Before the model runs, the cell prints the effective runtime and the operational ceilings — the `ResourceLimits` (rows, series, channels, windows), the fixed 512-step window and 8-step patch — and the device/precision policy. `validate_inputs` is the package's public validation stage: it runs exactly the two calls every task path makes (`validate_long_frame`, then `to_windows`), so it raises exactly what canonicalization would raise, and returns an **input manifest** naming the schema and ceilings, each window's series, valid positions, padding and truncation, the source-missingness fractions, and the verdict; it is written to `outputs/moment_embeddings_input_manifest.json`. To show what rejection looks like, the cell also validates a deliberately broken copy (a channel with no observed value) and records the pipeline's own error code as a finding. The canonical `WindowSet` used by the model is built by the same calls; any padding, truncation, source missingness, or irregular frequency is disclosed here before model execution. Successful output means the input passed validation and every canonicalization effect is disclosed before the model runs."
            ),
            "code": (
                'import os\n'
                '\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                'config = MomentConfig(task="embedding")\n'
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
                'with open("outputs/moment_embeddings_input_manifest.json", "w", encoding="utf-8") as handle:\n'
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
                '<details><summary>Check your reasoning</summary>One window: the package makes one canonical 512-step window per series and series `A` has 256 timestamps, so it is **left-padded** by 256 positions that the input mask excludes. The cell above prints `padded windows: 1 / 1` and `truncated windows: 0 / 1`; a series longer than 512 steps would instead keep only its final 512 timestamps and be flagged as truncated.</details>'
            ),
        },
        {
            "md": (
                '## 6. Extract embeddings\n'
                '\n'
                '`embed()` exercises the repository\'s supported embedding API on the verified pinned encoder (loaded in Section 3 with `task="embedding"`, whose head is `nn.Identity`). The embedding is **per window** (one 512-step canonical window per series), pooled with the `mean` reduction over patches, and every channel is embedded independently then averaged (`channel_policy`). MOMENT\'s upstream `embed` path has no per-point observedness mask: pre-filled missing positions are **visible to the encoder**, so the missing-data fractions must be interpreted alongside the vectors. Successful execution proves that this validated input can be processed by the verified pinned encoder and exposes the effective embedding contract; it does not establish downstream task quality.'
                '\n\n**Predict before running:** if a value were missing from your CSV, would the encoder ignore it? And which printed number would tell you the embedding is *good*?'
            ),
            "code": (
                'result = embed(windows, pipe, warmup=False)\n'
                'provenance = build_provenance(pipe, windows, result)\n'
                'print("effective model:", pipe.identity.name)\n'
                'print("effective revision:", pipe.identity.revision)\n'
                'print("verified weight file:", pipe.identity.weight_file_loaded)\n'
                'print("embedding shape:", result.embeddings.shape)\n'
                'print("reduction:", result.reduction)\n'
                'print("channel policy:", result.channel_policy)\n'
                'print("missingness visible to model:", result.missingness_visible_to_model)\n'
                'print("embedding L2 norm (sanity only):", float((result.embeddings[0] ** 2).sum() ** 0.5))'
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>No to the first: the upstream `embed` path has no per-point observedness mask, so pre-filled missing positions are **visible to the encoder** (`missingness visible to model`), and the missing fractions must be read beside the vectors. No to the second: nothing printed here measures quality. The L2 norm is a finiteness check; Section 7 reports `not-measurable`, and only a labelled downstream task (a linear probe, labelled retrieval) can measure a representation.</details>'
            ),
        },
        {
            "md": (
                '## 7. Evaluate → evaluation report\n'
                '\n'
                "There is **no intrinsic accuracy metric for an embedding vector**. Representation quality must be evaluated on a labelled or otherwise externally checkable downstream task appropriate to the use case — for example linear-probe classification, labelled retrieval precision, or a justified clustering validation design. `evaluation_report` is the package's public evaluation stage and always produces a report: for an `EmbeddingResult` the verdict is `not-measurable`, the metrics list is empty, and `needs` states what downstream labelled data would make the representations measurable. Successful output is a `not-measurable` report, which is the correct verdict for representations. The L2 norm printed above is only a finiteness/sanity check, not a quality score. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                'report = evaluation_report(result, model=pipe, sample_kind=sample_kind)\n'
                'with open("outputs/{stem}_evaluation_report.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(report, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(report, indent=2, default=str))\n'
                'if report["verdict"] == "not-measurable":\n'
                '    print("Embeddings are representations, not predictions; no correctness metric is computed.")'
            ),
        },
        {
            "md": (
                '## 8. Export outputs and provenance\n'
                '\n'
                "`outputs/moment_embeddings.csv` carries the input identifiers (`series_id`, `window_id`) alongside the vectors so identity survives downstream use; `outputs/moment_embeddings_provenance.json` carries the model/runtime/data provenance the package builds; `outputs/{stem}_result.json` carries the input manifest, the evaluation report, the sample identity and digest, the notebook's source (repository, revision, embedded module digests, generator), the model identifier, the immutable model revision and licence, and the runtime identity. Successful completion writes those files; file creation establishes a machine-readable handoff boundary and does **not** establish that the vectors are useful for a downstream task. No credentials are recorded."
            ),
            "code": (
                'embedding_frame = result.to_frame()\n'
                'embedding_frame.to_csv("outputs/moment_embeddings.csv", index=False)\n'
                'provenance["data"] = sample_identity\n'
                'provenance["notebook_source"] = NOTEBOOK_SOURCE\n'
                'with open("outputs/moment_embeddings_provenance.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(provenance, handle, indent=2, default=str)\n'
                'print(embedding_frame.iloc[:, :8])\n'
                'payload_out = {{\n'
                '    "result": {{"n_windows": len(result.window_ids), "embedding_shape": list(result.embeddings.shape), "reduction": result.reduction, "channel_policy": result.channel_policy, "missingness_visible_to_model": result.missingness_visible_to_model}},\n'
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
                'with open("outputs/moment_embeddings_result.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(payload_out, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(sorted(os.listdir("outputs")))'
            ),
        },
    ],
    "closing": (
        '## Interpretation and limits\n'
        '\n'
        "A successful run proves that the carried package can validate this input, resolve and integrity-check the pinned MOMENT checkpoint, execute the supported pooled-embedding path, and export identifier-preserving vectors with provenance. Successful execution proves that the recorded repository revision's package, carried in this notebook, can do exactly that — without the repository being reachable — and no more.\n"
        '\n'
        'It **does not prove** that these embeddings are accurate for classification, retrieval, clustering, forecasting, or any production domain; it also does not prove that missing values were ignored by the encoder. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain. Validate representation usefulness on a downstream task with domain-appropriate labelled evidence before deployment; the evaluation report says `not-measurable` because no such evidence exists here.\n'
        '\n'
        '**Next experiments:** compare downstream linear-probe or retrieval performance on clean versus missingness-bearing windows; evaluate task-specific representations on an independent labelled dataset; enable `USE_BYOD` with a multi-series CSV and inspect how padding and truncation are disclosed in the input manifest.\n'
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
        '- **Embedding / mean pooling** — the vector the encoder produces for a window, averaged over its patches (and here over channels); a representation, not a prediction.\n'
        '- **`not-measurable`** — the evaluation verdict when no ground truth exists, as for any embedding on its own.\n'
        '- **Linear probe** — a linear classifier trained on frozen embeddings; the cheapest honest test of a representation.\n'
        '- **BYOD** — bring your own data: your long-format CSV through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- The sample gave ___ window(s); padded ___ / truncated ___; the embedding shape was ___.\n'
        '- The evaluation verdict was ___ because ___.\n'
        '- To learn whether these vectors suit my task I would ___ (for example a labelled linear probe).\n'
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

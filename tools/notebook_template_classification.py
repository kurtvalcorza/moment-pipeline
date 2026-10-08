"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

This is the classification-adaptation (E2E) tutorial of moment-pipeline: the fourth notebook beside the
three TASK-INFERENCE ones (embeddings, imputation, anomaly scoring). It carries the same package plus the
two modules that hold the adaptation contract (`samples.py`, `adaptation.py`) and the same model cell
(`load_moment(task="embedding", weights_dir=WEIGHTS_DIR)`); MOMENT ships no classification head (RFC M-1/M-4),
so the head trained here is the notebook's own, over the pooled embeddings of the `embedding` task instance.

Workflow: the pinned snapshot is digest-verified and loaded, a digest-pinned real motion corpus (UCI HAPT,
CC BY 4.0) is fetched, validated and split by volunteer, the inference contract is exercised on real windows
through the repository's own validation / windowing / embedding path, the frozen embeddings are scored by a
majority floor, a cosine k-NN vote and a linear probe (the frozen policy), a bounded unfreeze of the last
encoder blocks is trained and selected against the probe on validation (the unfrozen policy), the held-out
split is scored, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "moment_pipeline",
    "repo_name": "moment-pipeline",
    "stem": "moment_classification",
    "notebook_name": "moment_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
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
    # MCL-m6: the measured size of the locked wheels (sum of the locked manylinux x86_64 wheel sizes from PyPI
    # release metadata, read on 2026-10-07; the lock's torch 2.14.0 is the CUDA build and carries the NVIDIA libraries).
    "lock_download": {"bytes": 3078108227, "measured": "sum of the locked manylinux x86_64 wheel sizes from PyPI release metadata, 2026-10-07; the lock's `torch==2.14.0` is the CUDA build, so the CPU path downloads its NVIDIA libraries too"},
    # MCL-m5: the upstream warning printed while loading is expected and refers to task heads, not the encoder.
    "model_section_note": "The loader prints upstream's warning *\"Only reconstruction head is pre-trained. Classification and forecasting heads must be fine-tuned.\"* — expected: the `task=\"embedding\"` instance replaces the head by an identity, and the classification head this notebook trains is its own (Sections 6–7), not upstream's untrained one.",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated hash-locked environment from the pinned dependencies, with nothing installed into the notebook kernel and no restart (including `momentfm` from "
        "its pinned source commit), stages and digest-verifies the pinned MOMENT-1-base snapshot (safetensors, 454 MB), "
        "fetches the 79.6 MB UCI HAPT archive (no credential), cuts 179 digest-pinned ten-second motion windows from it "
        "and draws 107 / 36 / 36 training, validation and test windows by a seeded split of whole volunteers, runs three "
        "test windows through the inference contract — the public validation, windowing and embedding path with an input "
        "manifest, a rejection probe and the `not-measurable` per-batch report — scores the frozen embeddings on the test "
        "split by a majority floor, a cosine 5-NN vote and a linear probe (the **frozen policy**), trains a bounded unfreeze "
        "of the last two encoder blocks with the head and selects between it and the probe by validation log-loss (the "
        "**unfrozen policy**), scores the held-out split with the selected model, prints predictions before and after, "
        "exports the head and any trained blocks as safetensors with a manifest, and reloads that artifact into a fresh "
        "pipeline to verify parity. The default path needs no repository clone, no DIMER worker or service, no credential, "
        "no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). Measured durations, each with its environment: the whole default path took 294 s on the recorded Kaggle Tesla T4 run of 2026-09-19 (downloads included) and 430 s of cell time on the local CPU pre-flight of 2026-09-19 (Windows, CPython 3.12, CPU float32, downloads pre-staged); the review of 2026-10-02 measured more than 590 s on another CPU. Treat any CPU figure as an estimate for your machine; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled windows as a `.zip` holding `records.csv` (columns `id`, `file`, `label`, `group`; `group` — the person, session or device — must be non-empty on every row, or the loader refuses the set) beside `.npy` "
        "arrays of shape `(channels, 512)` — arrays are decoded from the archive, never extracted to disk. They pass through "
        "the same validation, seeded group-disjoint split, floors, frozen-policy probe, unfrozen-policy training and "
        "selection, held-out evaluation, prediction, artifact export and reload-parity cells as the HAPT sample. The expected "
        "schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside this "
        "runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "LoadedMoment",
    "weights_key": "moment-1-base",
    "modules": [
        "adaptation.py",
        "anomaly.py",
        "canonical.py",
        "config.py",
        "csvio.py",
        "embedding.py",
        "imputation.py",
        "model.py",
        "provenance.py",
        "roles.py",
        "samples.py",
        "validation.py",
    ],
    "entry_module": "model.py",
    "model_load": 'load_moment(task="embedding", weights_dir=WEIGHTS_DIR)',
    "runtime_imports": ["torch", "transformers", "pandas", "numpy"],
    "title": "MOMENT-1-base — DIMER E2E supervised adaptation tutorial: activity classification from motion windows, linear probe vs bounded unfreeze (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/moment-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/moment-pipeline/blob/main/tutorials/moment_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutonLab%2FMOMENT--1--base-ffcc4d?style=flat",
            "https://huggingface.co/AutonLab/MOMENT-1-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-moment--timeseries--foundation--model%2Fmoment-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/moment-timeseries-foundation-model/moment",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2402.03885-b31b1b.svg", "https://arxiv.org/abs/2402.03885"),
    ],
    "capability": "pretrained time-series representation extraction (pooled 768-d window embeddings) and bounded supervised adaptation to a labelled-window classification task — a linear probe on the frozen embeddings with an optional unfreeze of the last encoder blocks — measured by held-out accuracy and macro-F1, using the pinned `AutonLab/MOMENT-1-base` encoder",
    "intro": (
        "At inference every 512-step window (each channel independently, 64 non-overlapping patches of 8 steps, "
        "instance-normalised per window) passes through the 12-block T5 encoder of MOMENT-1-base and the patch "
        "representations are mean-pooled into one 768-d vector per window; the `embedding` task instance replaces the "
        "pretrained reconstruction head with an identity, so the vector is a representation, not a prediction. The "
        "carried package adds manifest verification, the long-format validation and windowing contract, input "
        "validation with named ceilings, and the `validate_inputs`, `evaluation_report` and `build_provenance` "
        "helpers; the per-batch `evaluation_report` of an embedding is always `not-measurable`, because a vector has no "
        "intrinsic accuracy.\n\n"
        "What this notebook adds to inference is **supervised adaptation under an explicit frozen-vs-unfrozen policy** "
        "— the first task-head contract of this repository, and a caller-trained one: MOMENT-1-base ships no "
        "classification head and its upstream classification head is not pretrained. The dataset is real: 179 "
        "ten-second windows of smartphone motion data from the UCI *Smartphone-Based Recognition of Human Activities "
        "and Postural Transitions* dataset (HAPT, CC BY 4.0) — for each of 30 volunteers and each of six activities "
        "(walking, walking upstairs, walking downstairs, sitting, standing, laying) the centre 512 samples of the first "
        "labelled segment long enough to hold them, over six channels (accelerometer and gyroscope x/y/z); the 79.6 MB "
        "archive is pinned by byte size and SHA-256, fetched from the UCI repository at run time and read without "
        "extraction. Windows of one volunteer share a body and a phone, so the sample is split by **volunteer**, never "
        "by window. The carried `adaptation.py` scores predictions by **accuracy** and **macro-F1** with per-class "
        "recall and a confusion matrix; a **majority floor** and a **cosine 5-NN vote** over the frozen embeddings frame "
        "the numbers. The **frozen policy** trains only a linear head on the frozen embeddings (a linear probe); the "
        "**unfrozen policy** continues from that probe by training the last encoder blocks with the head end to end, and "
        "the epoch with the lowest validation log-loss — which may be the probe itself — is kept. The adaptation "
        "question is whether unfreezing buys anything over the probe on 107 training windows; the representation "
        "question underneath it is what per-window instance normalisation does to activities that differ mainly by "
        "gravity's direction. Nothing here is a quality claim about your series: it is one seeded split of one small "
        "corpus."
    ),
    "guided": {"opening": [(
        '**Who this notebook is for.** A learner who knows basic Python and pandas, has used Colab or Jupyter and has met time-series data, and wants to see how a pretrained time-series foundation model is used and adapted under an explicit frozen-vs-unfrozen policy — what goes in, what comes out, and what the output does and does not prove. The audience is students and practitioners preparing their own sensor or monitoring series; no prior experience with MOMENT is assumed — each term is explained where it first matters and again in the **Glossary**. CPU works (the unfreeze takes a few minutes); a T4 is faster.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | 179 ten-second, six-channel motion windows (512 samples at 50 Hz) cut from the digest-pinned UCI HAPT archive, split 107 / 36 / 36 by whole volunteers — or your own labelled windows |\n| Model | the MOMENT-1-base encoder (`task="embedding"`); a linear head on its frozen pooled embeddings (the probe, the frozen policy), optionally with the last two encoder blocks unfrozen (the unfrozen policy), selected on validation log-loss |\n| Output | an activity label and softmax probabilities per window, held-out accuracy and macro-F1 beside a majority floor and a 5-NN vote, and a safetensors adapter that reloads with identical probabilities |\n\n**How to use this notebook.** Choose a runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook\'s own Python, so no restart is needed (the recorded hosted runs of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the default path. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer. The worked answers quote the recorded Kaggle Tesla T4 run of 19 September 2026, named as such wherever a number appears; **CPU and CUDA runs of the same split can differ by a window or two** (the local CPU pre-flight of an earlier revision and the T4 run differed by one test window on the same split), so compare your printed comparison with the worked answer in windows, not in decimals. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the HAPT corpus, validation and a volunteer-level split *(evaluation practice: leakage)* → 5 the inference contract on real windows → 6 majority floor, 5-NN vote and the linear probe *(core concept: a frozen representation)* → 7 the bounded unfreeze, selected against the probe *(core concept: what is trained)* → 8 held-out evaluation *(evaluation practice)* → 9 before/after predictions, export and reload *(engineering)* → conclude.'
    )]},
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "model revision; fetch a digest-pinned real labelled corpus and validate and split it by volunteer without "
        "leakage; push real windows through the public validation, windowing and embedding path and read the input "
        "manifest, the vector contract and the `not-measurable` report correctly; read accuracy and macro-F1 beside a "
        "majority floor and a k-NN baseline; train a linear probe on frozen embeddings and a bounded unfreeze with "
        "explicit hyperparameters and validation-based selection between the two policies; evaluate on an independent "
        "volunteer-disjoint test split; compare predictions before and after; and export a safetensors adapter (head "
        "plus any trained blocks) that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "forecasting, anomaly decisions, imputation quality claims, full-encoder or patch-embedding training, data "
        "augmentation, any pretrained classification head, and any claim that six activities from 30 volunteers stand "
        "in for your series. The repository exposes none of these; the adapted encoder still emits representations and "
        "the trained head answers only for its six labels."
    ),
    "prerequisites": [
        '- **Learner:** basic Python and NumPy familiarity; no prior experience with MOMENT or time-series foundation models. Windows, patches, padding, masks and the evaluation verdicts are explained where they are first used and again in the Glossary.',
        '- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels (plus `momentfm`, built from its pinned commit), so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported.',
        "- **Compute:** the default path runs on CPU and uses CUDA automatically when available; the public API accepts `float32` only. One set of measured figures, environment named — the local CPU pre-flight of 2026-09-19 (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cpu`): the 179-window k-NN pass 18.7 s, the linear probe 76.8 s including its features, the three-epoch unfreeze of the last two blocks 307.5 s, the whole path 430 s; the review of 2026-10-02 saw more than 590 s on another CPU, and the recorded Kaggle Tesla T4 run of 2026-09-19 took 294 s for everything. The notebook prints its own timings, so read those rather than these estimates. The locked `torch==2.14.0` wheel (the CUDA build, ~555 MB) with its NVIDIA libraries and the 454 MB checkpoint are the large downloads of the run — see the measured total under External access — then the 79.6 MB archive.",
        "- **Knowledge:** basic Python, NumPy and pandas; what a long-format time-series table is; what a linear probe is and why it is the cheapest honest test of a representation; what accuracy and macro-F1 measure; what validation-based selection between two policies means.",
        "- **Data contract:** records are `{id, x, label}` — a float32 `(channels, 512)` array (or a path to a `.npy`) with 1..32 channels (an explicit `channels` list naming exactly that many unique channels, or the six HAPT names for six-channel windows and deterministic `channel_00`.. names otherwise — one ordered schema across the whole set), finite values of magnitude at most 1,000, a label of 1..64 plain characters, ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a training set needs 8..1,024 records and 2..100 classes with one channel count throughout; windows are de-duplicated by sample digest and split by `user` / `group` so one person's data never straddles splits. Every window then passes through the package's long-format validation (`series_id, timestamp, channel, value`) and canonical windowing exactly as inference input does. BYOD accepts a `.zip` (or a directory) holding `records.csv` and the `.npy` files.",
        "- **Validation is structural, not semantic:** nothing checks that a label is right for its window — a mislabelled set is trained on without complaint; the sample rate is assumed to be 50 Hz only for the timestamps the canonical path requires, and the model never sees it.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — wearable and motion recordings of identifiable people are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub snapshot, the default path fetches one pinned object — the HAPT archive at `https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip`, 79,596,192 bytes, SHA-256 `4ac4ae06…` in the carried `samples.py` — over HTTPS, refused on any byte-size or SHA-256 mismatch before it is opened; the dataset is CC BY 4.0 per the UCI repository's licence notice and is credited to its authors in the References.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and volunteer-level split\n\n"
                "`fetch_corpus` downloads the pinned archive (or reads it from the cache) and refuses a byte-size or "
                "SHA-256 mismatch before it is opened; `read_corpus` cuts the 179 windows out of the raw sensor files "
                "in memory — per volunteer and activity, the centre 512 samples of the first labelled segment at least "
                "512 samples long (one volunteer has no downstairs segment that long) — each a `{{id, x, label}}` record "
                "with its volunteer, experiment and sample offset. `build_sample_dataset` draws 18 training, 6 validation "
                "and 6 test **volunteers** by a seeded shuffle; `validate_dataset` then checks every record against the "
                "contract, `check_split_disjoint` asserts no window (by sample digest) and no volunteer appears in two "
                "splits, `user_summary` reports volunteers and label counts per split, and the training records table is "
                "written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 179 windows, six classes of 29–30, splits 107 / 36 / 36 with 18 / 6 / 6 volunteers, three "
                "digests, and four refusal probes — a duplicate id, a window of the wrong length, a non-finite window and a "
                "single-class dataset — each rejected before `torch` does anything. Success here means the corpus was "
                "fetched, verified, cut and split with no leakage; about a minute on the first run for the download."
                "\n\n**Predict before running:** why split by volunteer instead of by window? What would go wrong in the test score if one person's windows landed in both training and test?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import time\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = ""  # @param {{type:"string"}}\n'
                'SPLIT_SEED = 42  # @param {{type:"integer"}}\n\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                "t0 = time.perf_counter()\n"
                'def byod_file(path, kind, suffixes=()):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not source.is_file():\n'
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise RuntimeError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                '    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n'
                '        raise ValueError(f\'{{source.name}}: expected a {{kind}} ending in {{" or ".join(suffixes)}}.\')\n'
                '    return source\n\n\n'

                "if USE_BYOD:\n"
                "    byod_path = byod_file(BYOD_PATH, 'labelled-window archive or file accepted by load_byod_dataset')\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + byod_path.name + ')'\n"
                "    raw_count = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/hapt'))\n"
                "    raw_count = {{'windows': len(corpus), 'volunteers': len({{r['user'] for r in corpus}}), 'channels': list(CHANNELS)}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "fetch_seconds = round(time.perf_counter() - t0, 1)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "classes = class_names(train_records)\n"
                "disjoint = check_split_disjoint(splits)\n"
                "summary = user_summary(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw': raw_count, 'splits': disjoint, 'classes': classes, 'user_summary': summary, 'fetch_seconds': fetch_seconds, 'corpus_bytes': CORPUS_BYTES}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'users': manifest['users'], 'label_counts': manifest['label_counts'], 'n_channels': manifest['n_channels'], 'value_range': manifest['value_range'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{k: example[k] for k in ('id', 'label', 'user', 'experiment', 'first_sample', 'source') if k in example}}, 'shape': example['x'].shape}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'wrong window length': [{{**train_records[0], 'x': train_records[0]['x'][:, :-1]}}, *train_records[1:8]],\n"
                "    'non-finite window': [{{**train_records[0], 'x': train_records[0]['x'] * np.nan}}, *train_records[1:8]],\n"
                "    'single class': [{{**r, 'label': 'motion'}} for r in train_records[:8]],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Windows of one volunteer share a body, a phone and a way of moving, so a window-level split lets the model recognise the person instead of the activity and inflates the test score. The recorded run drew 18 / 6 / 6 volunteers (107 / 36 / 36 windows) and `check_split_disjoint` found no shared window or volunteer; all four refusal probes were rejected before the model ran.</details>'
            ),
        },
        {
            "md": (
                "## 5. Embed through the inference contract\n\n"
                "Before any adaptation, the inference contract is exercised as it always was, on three test windows. "
                "`records_to_long_frame` renders them as the `series_id, timestamp, channel, value` table every "
                "inference tutorial starts from; `validate_inputs` runs the package's `validate_long_frame` → `to_windows` "
                "path and returns an input manifest, while a deliberately broken frame — a duplicated "
                "(series, channel, timestamp) row — is validated too and its rejection recorded as a finding. `embed` "
                "returns one unit-norm 768-d vector per window, in window order, with `d_model`, `reduction` and the "
                "missingness fractions (0 here: the windows are complete); `evaluation_report` stays `not-measurable`, as "
                "it must for a batch of vectors, and `build_provenance` records the model, runtime and inference blocks. "
                "A first look at the representation is then printed, before any metric is read: the mean cosine similarity "
                "**within** each activity and **between** activities over the training windows (one table, not one pair). "
                "How to read it: pooled MOMENT vectors are all close in cosine — even a same-activity and a cross-activity "
                "pair can both score above 0.98, and the cross-activity pair can be the closer one, as the recorded Kaggle T4 "
                "run of 2026-09-19 showed (0.984 vs 0.997) — so a single pair is an anecdote; what carries information is "
                "whether the within-class mean sits *above* the between-class mean for a class, and by how much relative to "
                "the spread. **Cosine is a similarity, not a score**, and a vector carries no label; the k-NN vote in Section 6 "
                "is what turns these similarities into a prediction. Success here means the three stages ran and the four "
                "sanity checks are `True`; the training-split features take about 20 s on CPU and are computed here once."
            ),
            "code": (
                "probe_records = test_records[:3]\n"
                "config = MomentConfig(task='embedding')\n"
                "print({{'ceilings': {{'max_windows': config.limits.max_windows, 'max_channels': config.limits.max_channels, 'max_rows': config.limits.max_rows}}, 'contract': {{'sequence_length': config.sequence_length, 'patch_length': config.patch_length, 'd_model': pipe.identity.d_model_effective, 'ENCODER_BLOCKS': ENCODER_BLOCKS, 'task': pipe.identity.task}}}})\n"
                "frame = records_to_long_frame(probe_records)\n"
                "input_manifest = validate_inputs(frame, config, names=[r['id'] for r in probe_records])\n"
                "broken = pd.concat([frame, frame.iloc[:1]], ignore_index=True)\n"
                "try:\n"
                "    validate_inputs(broken, config)\n"
                "except ValidationError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'duplicate-row-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False, default=str)\n"
                "report, normalized = validate_long_frame(frame, config)\n"
                "windows = to_windows(normalized, config, report=report, frame=normalized)\n"
                "started = time.perf_counter()\n"
                "result = embed(windows, pipe, warmup=False)\n"
                "embed_seconds = round(time.perf_counter() - started, 3)\n"
                "vectors = np.asarray(result.embeddings, dtype=np.float32)\n"
                "vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)\n"
                "checks = {{\n"
                "    'one_vector_per_window': vectors.shape == (3, result.d_model) and result.d_model == pipe.identity.d_model_effective,\n"
                "    'window_order': list(result.series_ids) == list(windows.series_ids),\n"
                "    'complete_windows': result.masked_point_fraction == 0.0 and not any(result.padded),\n"
                "    'all_values_finite': bool(np.isfinite(vectors).all()),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'embed output failed a sanity check: {{checks}}')\n"
                "batch_report = evaluation_report(result, model=pipe, sample_kind='three HAPT test windows' if not USE_BYOD else 'three BYOD test windows')\n"
                "provenance = build_provenance(pipe, windows, result)\n"
                "ordered = {{sid: next(r for r in probe_records if r['id'] == sid) for sid in result.series_ids}}\n"
                "same = next((r for r in test_records[3:] if r['label'] == ordered[result.series_ids[0]]['label']), test_records[3])\n"
                "other = next((r for r in test_records[3:] if r['label'] != ordered[result.series_ids[0]]['label']), test_records[4])\n"
                "pair_vectors, _pair = features([same, other], pipe)\n"
                "print({{'probe_ids': list(result.series_ids), 'probe_labels': [ordered[s]['label'] for s in result.series_ids], 'seconds': embed_seconds, 'device': pipe.identity.device, 'checks': checks, 'findings': len(input_manifest['findings']), 'batch_report_verdict': batch_report['verdict'], 'per_point_missingness_mask_passed_to_model': result.missingness_visible_to_model}})\n"
                "print({{'cosine_same_activity': round(float(vectors[0] @ pair_vectors[0]), 4), 'cosine_other_activity': round(float(vectors[0] @ pair_vectors[1]), 4), 'note': 'one pair each; an anecdote - read the class table below instead'}})\n"
                "\n"
                "\n"
                "def class_cosine_table(vectors, labels):\n"
                "    \"\"\"Mean cosine within each class and between it and every other class, over unit-norm vectors.\"\"\"\n"
                "    vectors = np.asarray(vectors, dtype=np.float64)\n"
                "    labels = np.asarray(labels)\n"
                "    gram = vectors @ vectors.T\n"
                "    rows = {{}}\n"
                "    for name in sorted(set(labels.tolist())):\n"
                "        inside = labels == name\n"
                "        own = gram[np.ix_(inside, inside)]\n"
                "        off = own[~np.eye(own.shape[0], dtype=bool)]\n"
                "        other = gram[np.ix_(inside, ~inside)]\n"
                "        rows[name] = {{'n': int(inside.sum()), 'within': round(float(off.mean()), 4) if off.size else float('nan'), 'between': round(float(other.mean()), 4) if other.size else float('nan')}}\n"
                "        rows[name]['gap'] = round(rows[name]['within'] - rows[name]['between'], 4)\n"
                "    return rows\n"
                "\n"
                "\n"
                "t0 = time.perf_counter()\n"
                "train_vectors, train_ordered = features(train_records, pipe)\n"
                "cosine_table = class_cosine_table(train_vectors, [r['label'] for r in train_ordered])\n"
                "print({{'training_split_cosines': 'mean within-class vs between-class cosine over the training windows', 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "for name, row in cosine_table.items():\n"
                "    print({{name: row}})\n"
                "print({{'all_cosines_near_one': bool(min(r['between'] for r in cosine_table.values()) > 0.9), 'classes_with_positive_gap': [n for n, r in cosine_table.items() if r['gap'] > 0], 'note': 'a positive gap means windows of that activity sit closer to each other than to the rest; the size of the gaps, not the absolute cosines, is the signal'}})"
            ),
        },
        {
            "md": (
                "## 6. The majority floor, the k-NN baseline and the frozen policy\n\n"
                "Three numbers frame the adaptation, all on the 36 test windows. The **majority floor** predicts the "
                "most frequent training activity for every window — 1 / 6 here, since the sample is balanced. The "
                "**cosine 5-NN vote** labels each test window by its five nearest training windows in the frozen "
                "embedding space: what the representation gives with no training at all. The **frozen policy** is "
                "`adapt` with `trainable_blocks=0`: a linear head over the frozen, L2-normalised pooled embeddings, "
                "trained full-batch with AdamW for `PROBE_STEPS` steps — the linear probe — scored on validation as "
                "epoch 0 of its history and then on the test split by `evaluate` (accuracy, macro-F1, per-class recall, "
                "confusion). Read the per-class recall rather than the headline: in the recorded runs the three walking "
                "activities were separated almost perfectly while the three static postures were confused with each other, "
                "and the representation explains why — MOMENT instance-normalises every window and the postures differ "
                "mainly in the constant gravity component that normalisation removes. Success here "
                "means the frozen embeddings, k-NN and probe were scored and a verdict against the floor was printed (a probe that does not beat the floor "
                "is reported, not asserted, so a BYOD run still exports). The cell first restores the frozen encoder, so a re-run after an "
                "unfreeze scores the representation it is labelled with — this is the cell to re-run first whenever you change a form field in Section 6 or 7, or switch to BYOD in Section 4 (re-run from Section 4 then). Timing on the local CPU pre-flight of 2026-09-19: 18.7 s for the k-NN pass and 76.8 s for the probe with its features."
                '\n\n**Predict before running:** write down the test accuracy you expect from the majority floor, the 5-NN vote and the linear probe. Which activities will the probe confuse?'
            ),
            "code": (
                'PROBE_STEPS = 300  # @param {{type:"integer"}}\n'
                'PROBE_LR = 0.01  # @param {{type:"number"}}\n\n'
                '# SWP-F: the pinned encoder is snapshotted once, before any training, and restored here, so a re-run of this\n'
                '# section (or of Section 7) after an unfreeze won scores the frozen representation it is labelled with.\n'
                'import copy\n'
                '\n'
                '\n'
                'def encoder_digest(state):\n'
                '    digest = hashlib.sha256()\n'
                '    for name in sorted(state):\n'
                "        digest.update(name.encode('utf-8'))\n"
                '        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())\n'
                '    return digest.hexdigest()\n'
                '\n'
                '\n'
                "if '_FROZEN_ENCODER' not in globals():\n"
                '    _FROZEN_ENCODER = copy.deepcopy(pipe.pipeline.state_dict())\n'
                '    _FROZEN_ENCODER_DIGEST = encoder_digest(_FROZEN_ENCODER)\n'
                'pipe.pipeline.load_state_dict(_FROZEN_ENCODER, strict=True)\n'
                'pipe.pipeline.eval()\n'
                'if encoder_digest(pipe.pipeline.state_dict()) != _FROZEN_ENCODER_DIGEST:\n'
                "    raise RuntimeError('restoring the frozen encoder did not reproduce its digest')\n"
                "print({{'restarted_from_frozen_encoder': True, 'frozen_encoder_sha256': _FROZEN_ENCODER_DIGEST[:16] + '...'}})\n"
                '\n'
                "def brief(m):\n"
                "    return {{'accuracy': round(m['accuracy'], 4), 'macro_f1': round(m['macro_f1'], 4), 'n': m['n']}}\n\n"
                "floor = majority_baseline([r['label'] for r in train_records], [r['label'] for r in test_records], classes)\n"
                "print({{'majority_floor': brief(floor), 'baseline': floor['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "baseline_knn = knn_baseline(pipe, train_records, test_records, k=5)\n"
                "print({{'knn_baseline': brief(baseline_knn), 'baseline': baseline_knn['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "probe_adapter = adapt(pipe, train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=0)\n"
                "frozen_test = evaluate(pipe, probe_adapter, test_records)\n"
                "print({{'frozen_policy': probe_adapter.policy, 'probe_final_loss': round(probe_adapter.config['probe_final_loss'], 4), 'validation': probe_adapter.history[0]['val'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'frozen_policy_test': brief(frozen_test), 'log_loss': round(frozen_test['log_loss'], 4), 'verdict': frozen_test['verdict'], 'per_class_recall': {{c: round(v['recall'], 2) for c, v in frozen_test['per_class'].items()}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                'if probe_adapter.policy != POLICY_FROZEN:\n'
                "    raise RuntimeError(f'adapt with trainable_blocks=0 must return the frozen policy, got {{probe_adapter.policy!r}}')\n"
                "frozen_verdict = 'above the majority floor' if frozen_test['accuracy'] > floor['accuracy'] else 'not above the majority floor'\n"
                "print({{'frozen_policy_verdict': frozen_verdict}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the recorded Kaggle T4 run of 2026-09-19 the floor scored 16.7 % (one of six balanced classes), the 5-NN vote 72.2 % and the probe 72.2 % (macro-F1 0.708 and 0.715). The three walking activities were found perfectly and the static postures confused: laying and sitting recall 0.33 under the probe. MOMENT instance-normalises every window, which removes the constant gravity component that mostly distinguishes the postures. A CPU run can differ from these by a window or two (one window = 2.8 points).</details>'
            ),
        },
        {
            "md": (
                "## 7. The unfrozen policy: a bounded unfreeze selected against the probe\n\n"
                "`adapt` with `TRAINABLE_BLOCKS` > 0 first retrains the linear probe on the frozen embeddings (epoch 0 "
                "of the history, the frozen policy), then unfreezes the last `TRAINABLE_BLOCKS` T5 encoder blocks — two "
                "by default, 14,158,848 of 109,635,456 parameters; the patch embedding, the earlier blocks and the final "
                "norm stay frozen — and trains them with the head end to end on the windows for `EPOCHS` epochs (AdamW "
                "at `LEARNING_RATE`, weight decay 0.01, gradient clipping 1.0, seeded shuffling, no augmentation). Every "
                "epoch is scored on validation, and the epoch with the **lowest validation log-loss** is kept — epoch 0, "
                "the probe, competes on equal terms, so the selected policy can be either; the encoder inside `pipe` is "
                "modified in place only when the unfreeze wins. Accuracy and macro-F1 are printed beside the loss at "
                "every epoch.\n\n"
                "Watch the validation log-loss epoch by epoch: it need not fall monotonically, and the probe is kept whenever no epoch beats it. Success here means "
                "the ladder ran to its last epoch and named a selected policy and a `best_epoch`. Timing with its environment: 307.5 s on the local CPU pre-flight of 2026-09-19 (the whole T4 run took 294 s). **To re-run with other settings:** change `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE` or `TRAINABLE_BLOCKS` here, then re-run Section 6 and then Sections 7–9 in order; both cells restore the frozen encoder first, so each experiment starts from the pinned model and the frozen-policy numbers of Section 6 are reproduced exactly."
                '\n\n**Predict before running:** the unfreeze trains 14 M parameters on 107 windows. Will the validation log-loss fall every epoch, and will the unfreeze necessarily be selected over the probe?'
            ),
            "code": (
                'EPOCHS = 3  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 3e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 8  # @param {{type:"integer"}}\n'
                'TRAINABLE_BLOCKS = 2  # @param {{type:"integer"}}\n\n'
                "def report_epoch(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'stage': entry['stage'], 'train_loss': round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_log_loss'] = round(entry['val']['log_loss'], 4)\n"
                "        row['val_accuracy'] = round(entry['val']['accuracy'], 4)\n"
                "        row['val_macro_f1'] = round(entry['val']['macro_f1'], 4)\n"
                "    print(row)\n\n"
                '# SWP-F: start from the frozen encoder (snapshotted in Section 6), never from a previous unfreeze.\n'
                'pipe.pipeline.load_state_dict(_FROZEN_ENCODER, strict=True)\n'
                'pipe.pipeline.eval()\n'
                'if encoder_digest(pipe.pipeline.state_dict()) != _FROZEN_ENCODER_DIGEST:\n'
                "    raise RuntimeError('restoring the frozen encoder did not reproduce its digest')\n"
                '\n'
                "t0 = time.perf_counter()\n"
                "adapter = adapt(pipe, train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=TRAINABLE_BLOCKS, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, progress=report_epoch)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "report_epoch(adapter.history[0])\n"
                "print({{'selected_policy': adapter.policy, 'best_epoch': adapter.config['best_epoch'], 'selection': adapter.config['selection'], 'trainable_head': adapter.config['n_trainable_head'], 'trainable_blocks': adapter.config['n_trainable_blocks'], 'total_parameters': adapter.config['n_total'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Not necessarily on either count. With 107 training windows the validation log-loss can rise again after an epoch (it did in the recorded Kaggle T4 run, see Section 8), and the probe competes as epoch 0, so the unfreeze is kept only if one of its epochs beats the probe on validation log-loss; `selected_policy` and `best_epoch` say which won. The cell restored the frozen encoder first, so a re-run with another `TRAINABLE_BLOCKS` or `LEARNING_RATE` starts from the pinned model.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or policy selection, and no volunteer in it appears in the "
                "training or validation splits. The selected model is scored exactly as the frozen policy was in Section 6, "
                "and the four rows are put side by side: majority floor, k-NN vote, frozen policy, selected policy. Read "
                "the policy first: if validation kept the probe, the last two rows are the same model; if it chose the "
                "unfreeze, the delta is what the unfreeze bought on 36 windows. **What to notice:** the cell converts the accuracy delta into **windows** (`delta_windows = round(delta × n_test)`) and prints it beside the resolution of one window (1 / n_test ≈ 2.8 points here); a delta of zero to two windows is within what one seeded split of one small corpus can show by chance, and a per-class recall change of one window (0.17 on a six-window class) says nothing about a class on its own. Read the delta from *your* output; the worked answer below quotes the recorded run for comparison. The cell "
                "reports whether the selected model beats the majority floor (a verdict, not an assert, so a BYOD run still exports and reloads); it does **not** assert a gain over the probe, "
                "because that is the question, not the answer. 36 windows from six volunteers of one seeded split give "
                "no dispersion estimate, and CPU and CUDA runs of the same split can differ by a window. Success here means the comparison "
                "and the evaluation report were written."
                '\n\n**Predict before running:** if the unfreeze was selected, how many of the 36 test windows does a 3-point accuracy gain correspond to?'
            ),
            "code": (
                "adapted_test = evaluate(pipe, adapter, test_records)\n"
                "adapted_val = evaluate(pipe, adapter, val_records)\n"
                "comparison = {{\n"
                "    metric: {{'majority_floor': round(floor[metric], 4), 'knn5': round(baseline_knn[metric], 4), 'frozen_policy': round(frozen_test[metric], 4), 'selected_policy': round(adapted_test[metric], 4)}}\n"
                "    for metric in ('accuracy', 'macro_f1')\n"
                "}}\n"
                "comparison['log_loss'] = {{'frozen_policy': round(frozen_test['log_loss'], 4), 'selected_policy': round(adapted_test['log_loss'], 4)}}\n"
                "comparison['per_class_recall'] = {{c: {{'knn5': round(baseline_knn['per_class'][c]['recall'], 2), 'frozen': round(frozen_test['per_class'][c]['recall'], 2), 'selected': round(adapted_test['per_class'][c]['recall'], 2)}} for c in classes}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('accuracy', 'macro_f1')}}\n"
                "comparison['selected_policy'] = adapter.policy\n"
                "n_test = int(adapted_test['n'])\n"
                "delta_windows = int(round(comparison['delta_vs_frozen']['accuracy'] * n_test))\n"
                "comparison['delta_in_windows'] = {{'delta_windows': delta_windows, 'n_test': n_test, 'one_window_in_points': round(100.0 / n_test, 1), 'reading': 'the probe was kept: both rows are the same model' if adapter.policy == POLICY_FROZEN else ('within chance for one seeded split (0-2 windows)' if abs(delta_windows) <= 2 else 'more than two windows; still one split, no dispersion estimate')}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "print({{'confusion_selected': adapted_test['confusion'], 'classes': classes}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'user_summary': summary,\n"
                "    'classes': classes,\n"
                "    'batch_report': batch_report,\n"
                "    'baselines': {{'majority_floor': floor, 'knn5': baseline_knn}},\n"
                "    'frozen_policy': {{'adaptation': probe_adapter.summary(), 'history': probe_adapter.history, 'test': frozen_test}},\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': adapter.summary(),\n"
                "    'history': adapter.history,\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "    'verdicts': {{'frozen_policy_vs_floor': frozen_verdict, 'selected_policy_vs_floor': 'above the majority floor' if adapted_test['accuracy'] > floor['accuracy'] else 'not above the majority floor', 'selected_vs_frozen_accuracy_delta': comparison['delta_vs_frozen']['accuracy']}},\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "selected_verdict = 'above the majority floor' if adapted_test['accuracy'] > floor['accuracy'] else 'not above the majority floor'\n"
                "print({{'selected_policy_verdict': selected_verdict, 'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>A 3-point gain is one window (36 × 0.03 ≈ 1). In the recorded Kaggle T4 run of 2026-09-19 epoch 3 of the unfreeze was selected and scored 75.0 % accuracy and macro-F1 0.741 on the test split against the probe's 72.2 % and 0.715 — `delta_windows` 1 of 36 — with sitting recall 0.33 → 0.50 (one window) and laying unchanged at 0.33. That is within what one seeded split can show by chance, and no claim about the postures follows from it: whether unfreezing can recover information that per-window normalisation removed before the encoder is a hypothesis, and the test would be several split seeds (`SPLIT_SEED`) with the delta tabulated in windows. The verdict lines record the comparison instead of asserting it; the local CPU pre-flight of an earlier revision differed from the T4 run by one window on the same split.</details>"
            ),
        },
        {
            "md": (
                "## 9. Predict before and after, export the adapter and reload it\n\n"
                "Six test windows are labelled by `classify` with the selected model and printed beside the frozen "
                "policy's predictions (from the probe adapter of Section 6 over a fresh, untouched `load_moment` instance "
                "— after an unfreeze the encoder inside `pipe` has moved, so the frozen column needs its own encoder) and "
                "the gold activity with the top probability; read the probabilities as the head's softmax, not a "
                "calibrated confidence.\n\n"
                "`save_artifact` writes the head (`head.weight`, `head.bias`) and, when the unfrozen policy was selected, "
                "the trained block tensors — 18.6 KB for the head alone, about 56.7 MB with two trained blocks "
                "— as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the classes, the selected policy, the tensor "
                "names, the file size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`load_artifact` re-verifies the manifest and digest **before** deserialising, rebuilds the head from the "
                "manifest's classes, refuses any tensor that is not an encoder-block tensor of the base, and overlays "
                "the tensors onto a freshly loaded, digest-verified base — a new object from files, not the in-memory "
                "model (VER2). The cell asserts identical probabilities on the six windows and an identical test "
                "accuracy (VER4). Success here means the parity assertion passed and the six exports exist."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "show = test_records[:6]\n"
                "after = classify(pipe, adapter, show)\n"
                "frozen_pipe = load_moment(task='embedding', weights_dir=WEIGHTS_DIR)\n"
                "before = classify(frozen_pipe, probe_adapter, show)\n"
                "gold = {{r['id']: r for r in show}}\n"
                "rows = []\n"
                "for i, rid in enumerate(after['ids']):\n"
                "    rows.append({{'id': rid, 'gold': gold[rid]['label'], 'frozen_label': before['labels'][i], 'frozen_top_probability': round(max(before['probabilities'][i]), 4), 'selected_label': after['labels'][i], 'selected_top_probability': round(max(after['probabilities'][i]), 4), 'user': gold[rid].get('user', ''), 'source': gold[rid].get('source', '')}})\n"
                "    print({{k: rows[-1][k] for k in ('id', 'gold', 'frozen_label', 'frozen_top_probability', 'selected_label', 'selected_top_probability')}})\n"
                "print({{'selected_policy': after['policy'], 'decision_rule': after['decision_rule'], 'predictions_changed': sum(r['frozen_label'] != r['selected_label'] for r in rows), 'of': len(rows)}})\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "save_artifact(pipe, adapter, artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'policy': artifact_manifest['adapter']['policy'], 'classes': artifact_manifest['adapter']['classes'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded_pipe = load_moment(task='embedding', weights_dir=WEIGHTS_DIR)\n"
                "reloaded = load_artifact(reloaded_pipe, artifact_dir)\n"
                "reloaded_after = classify(reloaded_pipe, reloaded, show)\n"
                "reloaded_test = evaluate(reloaded_pipe, reloaded, test_records)\n"
                "parity = {{'probabilities_identical': reloaded_after['probabilities'] == after['probabilities'], 'accuracy_in_memory': round(adapted_test['accuracy'], 6), 'accuracy_reloaded': round(reloaded_test['accuracy'], 6), 'classes_identical': reloaded.classes == adapter.classes}}\n"
                "print({{'reload_parity': parity, 'reloaded_policy': reloaded.policy, 'reloaded_best_epoch': reloaded.config['best_epoch']}})\n"
                "assert parity['probabilities_identical'] and parity['classes_identical'] and abs(adapted_test['accuracy'] - reloaded_test['accuracy']) < 1e-9\n\n"
                "provenance['data'] = {{'kind': 'byod' if USE_BYOD else 'referenced-corpus', 'name': data_source, 'corpus_sha256': CORPUS_SHA256, 'corpus_bytes': CORPUS_BYTES, 'dataset_digests': evaluation_report_payload['dataset_digests']}}\n"
                "provenance['notebook_source'] = NOTEBOOK_SOURCE\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(MANIFEST['files']), 'total_bytes': MANIFEST['totalBytes'], 'fetched_this_run': fetched, 'weight_file': pipe.identity.weight_file_loaded, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.identity.weights_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'url': CORPUS_URL, 'bytes': CORPUS_BYTES, 'sha256': CORPUS_SHA256, 'license': CORPUS_LICENSE, 'windows': raw_count, 'activities': list(ACTIVITIES.values())}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_ids': list(result.series_ids), 'batch_report': batch_report, 'seconds': embed_seconds}},\n"
                "    'provenance': provenance,\n"
                "    'comparison': comparison,\n"
                "    'predictions_before_after': rows,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'policy': artifact_manifest['adapter']['policy']}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'pandas': pandas.__version__, 'numpy': numpy.__version__, 'device': pipe.identity.device, 'dtype': pipe.identity.dtype}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False, default=str)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The claim this notebook makes is about the **contract**, not about a number: the adaptation path runs both policies end to end "
        "on a real labelled corpus through the package's own validation and windowing path, chooses between them on "
        "validation rather than by assumption, and reports the answer against a floor and a no-training baseline rather "
        "than in isolation. **Read your own comparison** — the four accuracy rows, `delta_in_windows` and the per-class recall — "
        "and not a number from someone else's run: in the recorded Kaggle T4 run of 2026-09-19 the frozen embeddings put the 5-NN vote and the probe at 72.2 % against a floor of 16.7 %, "
        "the unfreeze was selected and scored 75.0 %, a delta of **one window** of 36, and a CPU pre-flight of an earlier revision landed one window away from that on the same split. A delta of that size is what one seeded split shows by chance.\n\n"
        "The test split is 36 windows from six volunteers of one seeded split of one small corpus with no dispersion "
        "estimate — one window is about 2.8 points of accuracy, so a two- or three-point delta is noise. The representation "
        "finding is the more useful one, and it is visible in both policies: MOMENT instance-normalises every window before patching, so the constant "
        "gravity component that tells sitting from standing from laying never reaches the encoder — the walking "
        "activities, which differ in dynamics, separate almost perfectly, while the static postures collapse into each "
        "other. Because the normalisation happens *before* the encoder, neither the head nor the unfrozen blocks can recover what it removed; a per-class recall that moves by one window after the unfreeze is not evidence that they did. Accuracy and macro-F1 "
        "say whether the gold activity is predicted, not whether the embeddings are good for any other task; the head's "
        "softmax is not a calibrated confidence. When the unfrozen policy is selected it changes the last blocks, which "
        "every window shares, so `embed` returns different vectors after it — cosines are not comparable across the "
        "frozen and adapted encoders, and the artifact records which policy won.\n\n"
        "Three things to carry to real data. **Floors first:** the majority floor and the k-NN vote on *your* windows are "
        "the numbers to read before any trained head's — if the probe barely beats k-NN, the representation already "
        "carries the task. **Leakage:** split by person, device or session (the contract splits by `user` / `group`, never "
        "by window). **Normalisation:** if your classes differ by level or offset rather than by shape, carry that "
        "information as an extra channel or a side feature — the encoder will not see it.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled "
        "corpus, validate the demonstrated dataset contract without leakage, execute the inference contract on real "
        "windows, a linear probe and a bounded unfreeze with validation-based policy selection, evaluate by accuracy and "
        "macro-F1 against a floor and a k-NN baseline on an independent split, and emit the shown machine-readable "
        "artifacts — without the repository being reachable. It does **not** establish benchmark superiority, "
        "representation quality on any other task, a usable acceptance threshold, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path; each one: Predict → change one field → re-run Section 6, then Sections 7–9 → observe → explain).** Set `LEARNING_RATE = 3e-5` and predict whether any epoch will beat the probe's validation log-loss (if none does, the probe is kept and Sections 8–9 report the same model twice; the artifact then holds the head only, and reload parity must still pass — it does, because Section 7 restored the pinned encoder before training). Set `LEARNING_RATE = 1e-4` and predict which epoch is selected. Set `TRAINABLE_BLOCKS = 1` or `4`, or `EPOCHS = 6`, and watch whether validation log-loss keeps falling or turns. Set another `SPLIT_SEED` in Section 4 and re-run from there: the floor, k-NN, probe and selected policy all move, which is the dispersion this one split cannot show. Drop the gyroscope channels in `read_corpus` and read what the accelerometer alone carries. Or bring your own labelled "
        "windows through BYOD (set `USE_BYOD = True` and `BYOD_PATH` in Section 4 and re-run from Section 4) and read the k-NN baseline before either policy. In every case the k-NN and frozen-policy numbers of Section 6 are computed on the pinned encoder, never on a previously adapted one.\n\n"
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **Section 1 fails while building `momentfm`** — the one source dependency is built from its pinned upstream commit and needs `git` and access to its Git host; run Section 1 again, or use Colab or Kaggle (both ship `git`).\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one …"** — set `BYOD_PATH` to a file in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **BYOD: a `validate_dataset` refusal** — it names the record and the rule (window length, channels, finite values, duplicate id, a single class); fix that record.\n'
        '- **CUDA out of memory in Section 7** — set `BATCH_SIZE = 4` or `TRAINABLE_BLOCKS = 1` and run Sections 6–9 again (numbers will differ from the recorded run).\n\n'
        '## Glossary\n\n'
        '- **Long-format table** — one row per `(series_id, timestamp, channel, value)`; the package turns it into windows.\n'
        '- **Window / patch** — the 512-step input MOMENT reads per series, cut into 64 non-overlapping 8-step patches.\n'
        '- **Left padding / truncation** — a series shorter than 512 steps is padded at the start and the padding masked out; a longer one keeps its final 512 timestamps.\n'
        '- **Instance normalisation** — each window is rescaled by its own mean and spread before the encoder, so the absolute level is removed.\n'
        '- **Input manifest** — the record of what validation saw and changed (padding, truncation, missingness) before the model ran.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **Volunteer-level split** — every window of one person goes to one split, so the test score is about unseen people.\n'
        '- **Majority floor / 5-NN vote** — always the most frequent training activity; the label of the five most similar training windows in embedding space.\n'
        '- **Linear probe (frozen policy)** — a linear head trained on frozen embeddings; **unfrozen policy** — the last encoder blocks trained with the head.\n'
        '- **Validation selection / log-loss** — the epoch with the lowest validation log-loss is kept, the probe (epoch 0) included.\n'
        '- **Accuracy / macro-F1 / per-class recall** — share correct; unweighted mean of per-class F1; share of each activity found.\n'
        '- **Adapter / reload parity** — the head and any trained blocks, overlaid on the pinned base; the reloaded model gives identical probabilities.\n'
        '- **Delta in windows** — the accuracy difference times the number of test windows; on 36 windows one window is 2.8 points, and a delta of 0–2 windows is within chance for one split.\n'
        '- **Within / between-class cosine** — mean similarity of an activity\'s windows to each other versus to the other activities; the gap, not the absolute value, is the signal.\n'
        '- **BYOD** — bring your own data: your labelled windows through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- Floor ___, 5-NN ___, probe ___, selected policy ___ (test accuracy); the selected policy was ___.\n'
        '- The activities confused most were ___, because ___.\n'
        '- One reason not to trust the gain over the probe yet: ___ (for example 36 test windows, one window ≈ 2.8 points).\n\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/moment-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/moment-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/moment-timeseries-foundation-model/moment\n"
        "- MOMENT: A Family of Open Time-series Foundation Models (Goswami et al., 2024): https://arxiv.org/abs/2402.03885\n"
        "- Reyes-Ortiz, Oneto, Samà, Parra, Anguita. Transition-Aware Human Activity Recognition Using Smartphones. Neurocomputing 171 (2016) — the HAPT dataset, UCI Machine Learning Repository id 341, CC BY 4.0: https://archive.ics.uci.edu/dataset/341\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}

# DIMER Model Card Specification

**Version 1.0 — 2026-09-09**

## 1. Purpose and scope

Every model pipeline built for DIMER ships a `MODEL_CARD.md`. This specification defines
the sections that card must contain and what each section must answer, so that cards
across pipelines are comparable, reviewable, and complete before a pipeline is released.

This specification applies to the model card only. Dataset cards, the DIMER pipeline
manifest, and the repository README are governed elsewhere.

The requirements here are a floor, not a ceiling. A card MAY carry any additional
sections it needs — supply-chain digests, input and output contracts, runtime
requirements, references — and existing cards SHOULD keep them.

## 2. Normative language

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be
interpreted as described in RFC 2119 as clarified by RFC 8174. Lowercase uses of these
words carry their ordinary English meaning.

A card that violates any **MUST** is not release-ready. A card that violates a **SHOULD**
is releasable if the author records the reason both in the pull request that ships it and
in the card itself. A pull request description stops being read the moment it merges; the
card is where a later reader looks.

## 3. General requirements

| ID | Requirement |
|---|---|
| **G1** | The card MUST open with a YAML front-matter block delimited by `---`, declaring at minimum `license` and `model_card_spec` — the version of this specification the card is written against, e.g. `model_card_spec: "1.0"`. Where the pipeline wraps an upstream model, the block MUST also declare `base_model` with the upstream model identifier. `model_card_spec` makes §8's fleet-wide coordination obligation checkable from the card itself, rather than by diffing copies of this document. |
| **G2** | The card MUST carry exactly one level-1 heading, naming the model and the packaged version. |
| **G3** | The card MUST contain every section listed in §4. Heading text is matched **case-insensitively**, ignoring surrounding emphasis: `Out-of-scope use cases` and `Out-of-Scope Use Cases` are the same section. §4 reproduces the DIMER template's own casing, which mixes Title Case and sentence case; a casing difference is not a nonconformance and MUST NOT be raised as one. |
| **G4** | Each required section MUST appear at the heading level given in §4. These levels reproduce the DIMER card template verbatim and are a **fidelity requirement, not a semantic hierarchy**: the block renders h1 → h6 → h4 → h6, which a table-of-contents generator, an accessibility linter, or GitHub's own outline will read as broken nesting. They are fixed so that every DIMER card renders identically against the same template. Correcting the nesting is a change to the template, and therefore a §8 change to this specification — never a per-card decision. |
| **G5** | Required sections MUST appear in the order given in §4, as one contiguous block. |
| **G6** | That block SHOULD sit directly after the title and any badge row, before repository-specific sections such as model details, provenance, or references. A reader reaching the technical detail should already have read the intended use and the limits. |
| **G7** | Each required section MUST be answered in the author's own prose. `<!-- Insert text here -->`, `TODO`, `TBD`, `FIXME`, and equivalent markers MUST NOT survive into a released card. |
| **G8** | A section MUST NOT be answered with a bare `N/A`, `None`, or `Not applicable`. Where a section genuinely does not apply, the card MUST say so **and say why**. "Not applicable in the demographic sense: this pipeline consumes machine telemetry with no human subjects" is an answer; `N/A` is not. |
| **G9** | Tooltip blockquotes (`> **Tooltip:** …`) are authoring aids. They MAY be kept in the card as guidance for later editors, but they do not constitute an answer, and a section that contains only its tooltip is unanswered. |
| **G10** | Every factual claim in a required section MUST be checkable against this repository — pinned revisions, recorded digests, the code path named, or a cited upstream paper. Claims about performance the pipeline does not measure MUST NOT appear. Where the pipeline does not measure something a section asks about, the card MUST state that it is not measured rather than estimate it. |
| **G11** | Each content section SHOULD run to at least 40 words. The binding criterion is the element list in §5, not the length; the floor exists because no listed element set can be discharged in a sentence fragment. |

## 4. Required sections

Container sections carry no prose of their own; their subsections carry it. A container
MAY carry a short framing sentence.

| Order | Heading | Level | Kind |
|---|---|---|---|
| 1 | `###### Description` | 6 | content |
| 2 | `#### Intended Use and Limitations` | 4 | container |
| 3 | `###### Primary Intended Uses` | 6 | content |
| 4 | `###### Primary Intended Users` | 6 | content |
| 5 | `###### Out-of-scope use cases` | 6 | content |
| 6 | `#### Factors` | 4 | container |
| 7 | `###### Groups` | 6 | content |
| 8 | `###### Instrumentation` | 6 | content |
| 9 | `###### Environment` | 6 | content |
| 10 | `#### Metrics` | 4 | container |
| 11 | `###### Performance Measures` | 6 | content |
| 12 | `###### Decision thresholds` | 6 | content |
| 13 | `###### Approaches to uncertainty and variability` | 6 | content |
| 14 | `#### Ethical considerations and biases` | 4 | container |
| 15 | `###### Data` | 6 | content |
| 16 | `###### Human Life` | 6 | content |
| 17 | `###### Mitigations` | 6 | content |
| 18 | `###### Risks and harms` | 6 | content |
| 19 | `###### Use cases` | 6 | content |

## 5. What each section must contain

Each subsection below gives the question the section answers, the elements a complete
answer contains, and the failure modes a reviewer rejects.

---

### 5.1 `###### Description`

**Answers:** what the model is and what idea it embodies.

The description MUST identify:

1. the upstream model and release packaged by this pipeline, by name and version;
2. the model family or architecture class, in terms a reader outside the project can follow;
3. the mechanism by which the model produces its output — what it does at inference time, and whether adaptation happens through training, in-context conditioning, or neither;
4. the boundary between the upstream model and this repository: what this repository adds — packaging, verification, a DIMER adapter — as distinct from what the upstream weights do.

**Rejected:** a paraphrase of the upstream README that never says what the pipeline itself
contributes; marketing adjectives (*state-of-the-art*, *powerful*, *cutting-edge*) in
place of a mechanism.

---

### 5.2 `#### Intended Use and Limitations`

**Answers:** the use cases envisioned during development.

This is a container. Its three subsections carry the content. The container MAY carry one
framing sentence.

---

### 5.3 `###### Primary Intended Uses`

**Answers:** the machine learning tasks the model is meant for.

The section MUST identify:

1. the task type in technical terms — the input it takes and the output it produces;
2. concrete application domains envisioned during development, named specifically enough that a reader can tell whether their own problem is one of them;
3. the role the pipeline is meant to play in a larger system, where that is a design intent — for example a strong zero-configuration baseline, a feature extractor feeding a downstream model, or a production inference service.

Uses MAY be defined as broadly or as narrowly as the developers intended. A narrow
statement of intent is preferred to a broad one the pipeline cannot support.

**Rejected:** a task type alone with no application domain; a domain list so broad it
excludes nothing.

---

### 5.4 `###### Primary Intended Users`

**Answers:** who the model was built for.

The section MUST identify:

1. the intended user roles — for example machine learning engineers, data scientists, domain analysts, application developers;
2. the deployment setting envisioned: research, internal enterprise, public service, hobbyist;
3. the competencies the pipeline assumes of its users — what a user is expected to already understand in order to use it safely. This is what tells a reader how robust the pipeline is to the inputs they will give it.

**Rejected:** "anyone"; a role list with no assumed competencies, which leaves the
robustness question unanswered.

---

### 5.5 `###### Out-of-scope use cases`

**Answers:** what this model is not for.

The section MUST enumerate, as a list:

1. **capability boundaries** — tasks the model architecture cannot perform, and where applicable the sibling pipeline or service that does perform them;
2. **input boundaries** — data shapes, modalities, sizes, or distributions the pipeline rejects or degrades on, with the operative limit stated numerically where one exists;
3. **decision boundaries** — deployments the developers rule out on consequence grounds, in particular autonomous or high-impact decisions taken without human oversight.

Each entry SHOULD read as a disclaimer a user could act on, in the manner of a warning
label: "not for use on text examples shorter than 100 tokens" is actionable; "may not
work well on some inputs" is not.

**Rejected:** an empty or one-line section. Every model has out-of-scope uses; a card that
lists none has not looked.

---

### 5.6 `#### Factors`

**Answers:** what the model's behaviour varies with.

This is a container covering demographic and phenotypic groups, instrumentation, and
environmental conditions considered during development. Its three subsections carry the
content.

---

### 5.7 `###### Groups`

**Answers:** which distinct categories of data instance the model's behaviour was
considered against.

The section MUST state:

1. whether the pipeline is human-centric, and on what basis;
2. for a human-centric pipeline: the groups present in the evaluation data — people sharing one or more characteristics, such as age, gender, skin type, or socioeconomic status — and any group-level performance differences observed;
3. for a pipeline that is not human-centric, or whose upstream pretraining corpus is not group-audited: an explicit statement of that fact, and the obligation it transfers to the downstream operator, naming the fairness audit the operator is expected to perform on their own data.

**Rejected:** `N/A`; silence about an unaudited pretraining corpus. Where group structure
is unknown, the unknown MUST be stated, not omitted.

---

### 5.8 `###### Instrumentation`

**Answers:** what captured the data the model consumes.

The section MUST identify:

1. the instruments or systems that produce the training and evaluation data — sensors, cameras, assays, survey instruments, transactional databases, ETL pipelines;
2. the instrument characteristics that materially affect the data — sampling rate, resolution, calibration, encoding;
3. how instrument error propagates: which upstream defects — drift, miscalibration, changed collection procedure — reach the model as feature error, and whether the pipeline can detect them.

Where the model consumes an abstract numeric representation rather than a raw sensor
stream, the section MUST say so and identify the data-producing systems anyway. The
instrument does not disappear because a table sits between it and the model.

---

### 5.9 `###### Environment`

**Answers:** the conditions under which the reported behaviour holds.

The section MUST cover both readings of *environment*:

1. **operating environment** — the compute the pipeline runs on and the constraints attached to it: supported devices, accelerator requirements, precision, and any feature that is unavailable on some hardware;
2. **data environment** — the conditions the deployment data is assumed to satisfy, such as the distributional relationship assumed between training or support data and inference data, physical capture conditions where the data is sensed, and the degradation expected when those assumptions fail.

**Rejected:** hardware alone. The distributional assumption is the half that determines
whether the model's output means anything.

---

### 5.10 `#### Metrics`

**Answers:** how the model's real-world impact is measured.

This is a container. Metrics MUST be chosen for the model's structure and intended use,
and the subsections MUST justify that choice rather than list defaults.

---

### 5.11 `###### Performance Measures`

**Answers:** what is reported, and why these measures and not others.

The section MUST state:

1. each performance measure the pipeline reports, named exactly as the code reports it;
2. what each measure captures — discrete correctness, ranking quality, probability calibration, reconstruction error;
3. the justification for the selection: why these measures suit this model's structure and intended use, and what a reader would lose by reading only one of them;
4. where the pipeline reports no performance measure — because it emits an unlabelled score or a representation rather than a prediction — a statement of that fact and of what the caller must supply to evaluate it.

**Rejected:** a metric list with no justification. The question is comparative: why these
measures over the alternatives.

---

### 5.12 `###### Decision thresholds`

**Answers:** every threshold applied in developing the model, and every one left to the
caller.

The section MUST state:

1. the default decision rule the pipeline applies, including an implicit one — an `argmax` over class probabilities is a threshold and MUST be named as such;
2. any acceptance threshold set during development, with its value and the reason for that value, including thresholds imposed by the target domain;
3. thresholds deliberately not shipped: where the pipeline emits an unthresholded score, the card MUST state that no threshold is applied, why one was withheld, and who owns calibrating it;
4. the guidance a deployment needs to set its own threshold, in terms of the asymmetric cost of false positives against false negatives.

**Rejected:** "the default threshold is 0.5" with no statement of who should change it and
on what basis; silence about an implicit `argmax`.

---

### 5.13 `###### Approaches to uncertainty and variability`

**Answers:** how the reported numbers were estimated, and how much they move.

The section MUST state:

1. the estimation procedure behind any reported metric — a single holdout split, an average over *n* runs, *k*-fold cross-validation, bootstrap resampling — with the number of runs or folds;
2. the dispersion reported alongside the central value, if any: standard deviation, variance, confidence interval, or the absence of one;
3. the sources of run-to-run variability in the pipeline — sampling, ensembling, seeding, non-deterministic kernels — and which of them are controlled by a seed;
4. the status of any confidence output: whether a probability the pipeline emits is calibrated, and if it is not, what the caller must do to obtain a calibrated one;
5. where the pipeline reports no metric at all — because it emits a representation or a raw, uncalibrated score rather than a prediction — a statement of that fact, and of what the caller must supply to estimate one. This mirrors §5.11 item 4: a pipeline with nothing to report says so, and does not manufacture a procedure for a number it does not produce.

**Rejected:** reporting a metric with no estimation procedure; presenting an uncalibrated
softmax output as a probability without qualification.

---

### 5.14 `#### Ethical considerations and biases`

**Answers:** the ethical considerations that went into development, and what was done
about them.

This is a container. Where the model or pipeline was reviewed by an external board, or
cleared through testing with a specific group, that review MUST be recorded here or in
`Mitigations`, with its scope and outcome. Where no such review took place, the card MUST
NOT imply that one did.

---

### 5.15 `###### Data`

**Answers:** whether the model uses sensitive data.

The section MUST state:

1. what the model was pretrained or trained on, at the level of detail the upstream authors disclose, and where that disclosure ends;
2. whether any of it is sensitive — personal, classified, proprietary, or otherwise restricted — and whether that is known or merely not ruled out;
3. what this repository distributes: weights, code, sample data, and explicitly what it does not distribute;
4. the operator's obligation for the data they supply at inference — the audit for personal, sensitive, or proprietary attributes that the pipeline does not perform for them.

**Rejected:** "no sensitive data" asserted for an upstream corpus the authors did not
enumerate. Where sensitivity is unknown, say it is unknown.

---

### 5.16 `###### Human Life`

**Answers:** whether the model bears on matters central to human life or flourishing.

The section MUST state:

1. whether the pipeline is intended for decisions in domains such as health, safety, criminal justice, employment, credit, or housing;
2. the certification and validation status behind that answer — what the pipeline has and has not been validated for, and by whom;
3. where use in a sensitive domain is foreseeable even though it is not intended, the conditions under which it would be admissible: human oversight, independent clinical or domain validation, regulatory clearance.

A negative answer MUST be stated as a negative, not implied by omission.

---

### 5.17 `###### Mitigations`

**Answers:** what was actually done to reduce risk.

The section MUST list mitigations implemented in this repository, each traceable to
something a reviewer can inspect. Where they apply, it MUST cover:

1. **supply-chain integrity** — pinned upstream revisions, recorded digests, verification performed at load time, and refusal of unpinned or unverified sources;
2. **input integrity** — schema validation, rejection of malformed or misaligned input, handling of missing and unseen values;
3. **statistical mitigations** — class balancing, subsampling policy, or equivalent measures against known failure modes;
4. **reproducibility** — seed control, locked dependency graphs, recorded provenance;
5. **refusals** — capabilities the pipeline deliberately declines to expose, and how the refusal is enforced.

Each entry MUST name the mechanism, not the intention. "Inputs are validated" is an
intention; "inference input must match the fitted feature names exactly, and a mismatch
is rejected" is a mechanism.

**Rejected:** aspirational mitigations that are not implemented. A planned control belongs
in the repository's roadmap, not in this section.

---

### 5.18 `###### Risks and harms`

**Answers:** what can go wrong in use.

The section MUST enumerate the risks arising from use of the model. Each entry SHOULD
identify:

1. the failure mode, in mechanical terms;
2. who bears the harm — the operator, the data subject, a third party;
3. the conditions under which the risk is realised, and its likelihood under normal use where that can be characterised;
4. the magnitude of the resulting harm.

The enumeration MUST include model-intrinsic risks — overconfidence outside the training
distribution, amplification of bias present in the input data — and use-context risks such
as automation bias and undetected data leakage.

**Rejected:** a risk list that names only harms the model would cause on its own, omitting
those created by how it will be used.

---

### 5.19 `###### Use cases`

**Answers:** which uses of this model are disturbing or prohibited.

The section MUST enumerate uses the developers consider unacceptable, distinct from the
capability and decision boundaries already listed in `Out-of-scope use cases`. §5.5 says
what the model cannot or should not do; this section says what it MUST NOT be used for
even where it would work. Cover, where applicable:

1. surveillance, biometric or demographic profiling, and social scoring;
2. unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access;
3. deceptive, manipulative, or predatory applications;
4. any use prohibited by the upstream model licence or by the terms of the deployment.

Where the developers identify no such use, the card MUST say so explicitly and give the
reasoning. Silence is not an answer.

---

## 6. Pre-flight checklist

Run this before opening a release pull request, and again as a reviewer. Every line is a
**MUST** unless marked otherwise.

**Structure**

- [ ] Front matter present, declaring `license`, and `base_model` where an upstream model is wrapped (G1).
- [ ] Exactly one level-1 title, naming model and version (G2).
- [ ] All 19 sections of §4 present, spelled as given (G3).
- [ ] Every heading at its specified level (G4).
- [ ] Sections in the specified order, as one contiguous block (G5).
- [ ] Block positioned ahead of repository-specific sections (G6, SHOULD).

**Content**

- [ ] No placeholder or marker text anywhere in the block (G7).
- [ ] No section answered with a bare `N/A`, `None`, or `Not applicable` (G8).
- [ ] No section consisting only of its tooltip (G9).
- [ ] Every element list in §5 discharged, section by section.
- [ ] Every factual claim checkable against the repository or a cited source (G10).
- [ ] No performance claim the pipeline does not measure (G10).
- [ ] No mitigation listed that is not implemented (§5.17).
- [ ] No implied external review that did not occur (§5.14).

**Consistency**

- [ ] The card's stated capabilities match what the code exposes; a capability the loader refuses is described as refused.
- [ ] Pins, digests, and version strings in the card match the values in the code.
- [ ] Metric names match the identifiers the pipeline reports.
- [ ] Out-of-scope uses and prohibited uses do not contradict the intended uses.

## 7. Template

Copy this block into a new pipeline's `MODEL_CARD.md`, directly after the title and badge
row, and replace each placeholder. Tooltips MAY be kept; the placeholder comments MUST NOT
survive (G7).

```markdown
###### Description

<!-- Insert text here -->

> **Tooltip:** Discuss clearly the idea of your model.

#### Intended Use and Limitations

> **Tooltip:** Provide use cases that were envisioned during development.

###### Primary Intended Uses

<!-- Insert text here -->

> **Tooltip:** Describe the intended general or specific machine learning tasks in mind. Use cases may be as broadly or narrowly defined as the developers intended. For example, if the model was built simply to label images, then this task should be indicated as the primary intended use case.

###### Primary Intended Users

<!-- Insert text here -->

> **Tooltip:** Describe the primary intended users of the model. This helps users gain insight into how robust the model may be to different kinds of inputs. For example, was the model developed for entertainment purposes, for hobbyists, or enterprise solutions?

###### Out-of-scope use cases

<!-- Insert text here -->

> **Tooltip:** Describe possible usages of the model that are outside the scope intended by its developers. This is inspired by warning labels on food and toys, and similar disclaimers presented in electronic datasheets. For example, "not for use on text examples shorter than 100 tokens".

---

#### Factors

> **Tooltip:** This section describes the demographic or phenotypic groups, environmental conditions, technical attributes, and instrumentation conditions that were considered during model development.

###### Groups

<!-- Insert text here -->

> **Tooltip:** Provide the distinct categories with similar characteristics that are present in the evaluation of data instances. For a human-centric machine learning model, "groups" are people who share one or multiple characteristics. For human-centric computer vision models, the visual presentation of age, gender, and skin type may be relevant.

###### Instrumentation

<!-- Insert text here -->

> **Tooltip:** Provide information about the instruments that were used to capture the training and evaluation datasets for model development. For example, for a face detection model, the instruments may include the type of camera used, type of lens, image stabilization method or dynamic range techniques used in the camera's software.

###### Environment

<!-- Insert text here -->

> **Tooltip:** Provide information about the model's performance with respect to the different environmental settings considered in the development of the model. For example, environment settings may include level of humidity, amount of light, and type of light in the environment in which training and evaluation datasets have been captured.

---

#### Metrics

> **Tooltip:** This section reflects potential real-world impacts of the model. Metrics should be determined based on the model's structure and intended use.

###### Performance Measures

<!-- Insert text here -->

> **Tooltip:** What measures of model performance are being reported, and why were they selected over other measures of model performance?

###### Decision thresholds

<!-- Insert text here -->

> **Tooltip:** Cite all the decision thresholds that were applied in developing the model. For example, the accuracy threshold that was set for the model was at least of 95% accuracy due to its target application in the medical domain.

###### Approaches to uncertainty and variability

<!-- Insert text here -->

> **Tooltip:** How are the measurements and estimations of these metrics calculated? For example, this may include standard deviation, variance, confidence intervals, or KL divergence. Details of how these values are approximated should also be included (e.g., average of 5 runs, 10-fold cross-validation).

---

#### Ethical considerations and biases

> **Tooltip:** Information about the ethical considerations that went into model development, surfacing ethical challenges and solutions to stakeholders. Include other ethical considerations such as review by an external board or testing with a specific group for clearing.

###### Data

<!-- Insert text here -->

> **Tooltip:** Does the model use any sensitive data. For example, classified data.

###### Human Life

<!-- Insert text here -->

> **Tooltip:** Is the model intended to perform in a situation for decision-making in matters that are central to human life or flourishing. For example, criminal sentence, health, and safety.

###### Mitigations

<!-- Insert text here -->

> **Tooltip:** What were the risk mitigations employed during model development?

###### Risks and harms

<!-- Insert text here -->

> **Tooltip:** What risks may occur when using the model? Identify potential recipients of risks, likelihood of risks, and magnitude of harms brought by the risks.

###### Use cases

<!-- Insert text here -->

> **Tooltip:** Are there any model use cases that are considered as disturbing?
```

## 8. Change control

This specification is versioned and shared across DIMER pipeline repositories. A change to
§4 or §5 changes the contract every pipeline is held to, and therefore:

- MUST be applied to every DIMER pipeline repository that carries this document;
- MUST increment the version in the header — a new or removed required section, or a new
  mandatory element, is a major increment; clarified wording is a minor one;
- SHOULD be accompanied by an assessment of which existing cards it puts out of
  conformance, so that the work is visible rather than discovered later.

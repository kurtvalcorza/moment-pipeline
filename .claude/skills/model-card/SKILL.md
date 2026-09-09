---
name: model-card
description: Author, review, or pre-flight a DIMER pipeline's MODEL_CARD.md against the DIMER Model Card Specification. Use when writing a model card for a new pipeline, adding the required sections to an existing card, reviewing a card before release, or answering what a DIMER model card must contain. Triggers on "model card", "MODEL_CARD.md", "model card review", "pre-flight the card", "does the card conform".
---

# DIMER model card authoring and review

`MODEL_CARD_SPEC.md` at the repository root is the contract. It defines 19 required
sections, their heading levels, their order, and the elements each section must answer.
**Read it before writing or reviewing a card.** Do not work from memory of the section
list, and do not restate the spec here — quote it and follow it.

## Authoring a card

1. Read `MODEL_CARD_SPEC.md` §4 and §5.
2. Read the repository before writing a word of the card. The card's claims must come from
   the code: the loader and what it refuses, pinned revisions and digests, the metrics the
   evaluation path actually reports, the validation the input path performs, the seeds and
   locks that make a run reproducible, and the RFC or design doc if the repository carries
   one. A card written from the upstream README will assert things this pipeline does not do.
3. Copy the §7 template into place, positioned per **G6** — after the title and badge row,
   before repository-specific sections such as model details, provenance, and references.
4. Answer each section against its element list in §5. The element list is the acceptance
   criterion; the word floor in **G11** is not.
5. Run the §6 pre-flight checklist over your own draft before proposing it.

## Reviewing a card

Work the §6 checklist in order — structure, then content, then consistency — and report by
section with line numbers. For each finding give the section, the requirement ID or §5
element that is unmet, and what would satisfy it. A review that says a section is "thin"
without naming the missing element is not actionable.

Check the card against the code, not only against itself. The recurring defects are:

- a capability described in the card that the loader refuses, or a capability the code
  exposes that the card never mentions;
- pins, digests, or version strings that have drifted from the values in the source;
- metric names in the card that no longer match the identifiers the pipeline reports;
- mitigations that describe an intention rather than an implemented mechanism;
- performance claims for something the pipeline does not measure.

## Rules that decide most findings

- **Say what is not known.** Where an upstream corpus was not audited, where a probability
  is uncalibrated, where no threshold is shipped, where no external review took place — the
  card states it. Omission reads as a claim.
- **A bare `N/A` is never an answer** (**G8**). "Does not apply, because …" is.
- **Tooltips are not content** (**G9**). A section holding only its tooltip is unanswered.
- **Mechanism over intention** (§5.17). Name the check, the pin, the refusal — not the goal.
- **`Out-of-scope use cases` and `Use cases` are different questions.** §5.5 is what the
  model cannot or should not do; §5.19 is what it must not be used for even where it works.

## Changing the specification

`MODEL_CARD_SPEC.md` §8 governs. A change to the required sections binds every DIMER
pipeline: apply it to each repository carrying the document, increment the version, and
say which existing cards it puts out of conformance.

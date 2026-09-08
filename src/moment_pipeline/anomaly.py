"""Task 3 — reconstruction-based anomaly scoring. **Deferred to Phase 4.**

This module is intentionally empty of behaviour. The RFC repository layout names it, and
leaving a working-looking stub here would be worse than leaving nothing: a partially
implemented residual score invites callers to treat it as calibrated.

What Phase 4 owes, per the RFC:

* the primitive score is a per-element reconstruction residual with shape
  `(batch, channel, timestep)` under a documented loss (absolute or squared);
* raw scores are mandatory output; there is **no universal binary threshold** in v1;
* channel aggregation (max / mean / none) is DIMER-owned, explicit, and recorded in
  metadata, with "no aggregation" the default;
* the score is an *unmasked self-reconstruction residual*: it is not a forecast residual
  and gives no guarantee of sensitivity to drift the model reconstructs easily.

Phase 1 ships only the reconstruction primitive those scores would be built on —
`moment_pipeline.imputation.reconstruct`.
"""

from __future__ import annotations

__all__: list[str] = []

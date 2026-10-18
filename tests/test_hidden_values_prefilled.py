"""Review m3 (unit, no weights): positions MOMENT is told to hide reach the model as the pre-fill
sentinel."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import torch

from moment_pipeline.canonical import to_windows
from moment_pipeline.config import MomentConfig
from moment_pipeline.imputation import reconstruct
from moment_pipeline.validation import validate_long_frame


class _Recorder:
    def __init__(self):
        self.seen = []

    def reconstruct(self, x_enc, input_mask, mask):
        self.seen.append(x_enc.clone())
        return types.SimpleNamespace(reconstruction=torch.zeros_like(x_enc))


def test_hidden_positions_reach_the_model_as_prefill_and_the_truth_is_kept():
    stamps = pd.date_range("2026-01-01", periods=64, freq="15min")
    frame = pd.DataFrame(
        {
            "series_id": "S",
            "timestamp": stamps,
            "channel": "c",
            "value": np.arange(64, dtype=float) + 5,
        }
    )
    config = MomentConfig(task="reconstruction")
    report, normalized = validate_long_frame(frame, config)
    windows = to_windows(normalized, config, report=report, frame=normalized)
    recorder = _Recorder()
    model = types.SimpleNamespace(
        identity=types.SimpleNamespace(task="reconstruction", device="cpu"), pipeline=recorder
    )
    visible = np.ones_like(windows.input_mask, dtype=np.float32)
    visible[0, 496:504] = 0
    result = reconstruct(windows, model, mask=visible, warmup=False)
    seen = recorder.seen[0].numpy()
    hidden = result.model_mask[0] == 0
    assert np.all(seen[0, :, hidden.nonzero()[0]] == windows.prefill_value)
    assert np.array_equal(seen[0, :, ~hidden], windows.x_enc[0, :, ~hidden])
    assert np.array_equal(result.input_values, windows.x_enc), (
        "the scored truth is the caller's x_enc, untouched"
    )
    assert windows.x_enc[0, 0, 500] != windows.prefill_value

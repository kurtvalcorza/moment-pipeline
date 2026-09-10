"""Strict CSV ingestion for long-format DIMER notebook input.

The notebook specification requires duplicate or otherwise ambiguous CSV headers to
be rejected before pandas can silently mangle them. This module owns that raw-byte
boundary; callers should still pass the returned frame through ``validate_long_frame``
for the production data contract.
"""

from __future__ import annotations

import csv
import io

import pandas as pd

from .validation import REQUIRED_COLUMNS, ValidationError


def read_long_csv_bytes(payload: bytes) -> pd.DataFrame:
    """Parse UTF-8 CSV bytes only after validating the raw header.

    Header comparison is case-insensitive and ignores surrounding whitespace for
    ambiguity detection, while the actual required column names remain exact. This
    prevents inputs such as ``value,value`` or ``value, Value `` from being silently
    renamed by pandas before the repository validator can inspect the schema.
    """
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(
            "INVALID_CSV_ENCODING",
            "CSV input must be UTF-8 (an optional UTF-8 BOM is accepted)",
        ) from exc

    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        raise ValidationError("EMPTY_CSV", "CSV input has no header row")

    canonical = [column.strip().casefold() for column in header]
    blank_positions = [index for index, column in enumerate(canonical) if not column]
    if blank_positions:
        raise ValidationError(
            "AMBIGUOUS_COLUMNS",
            "CSV header contains blank column name(s)",
            {"column_positions": blank_positions},
        )

    duplicates = sorted(
        {canonical[index] for index, column in enumerate(canonical) if canonical.count(column) > 1}
    )
    if duplicates:
        raise ValidationError(
            "DUPLICATE_COLUMNS",
            "CSV header contains duplicate or ambiguous column names before dataframe parsing: "
            f"{duplicates}",
            {"columns": duplicates},
        )

    present = set(header)
    missing = [column for column in REQUIRED_COLUMNS if column not in present]
    if missing:
        raise ValidationError(
            "MISSING_COLUMNS",
            f"long-format input requires columns {list(REQUIRED_COLUMNS)}; missing {missing}",
            {"missing": missing, "present": header},
        )

    return pd.read_csv(io.StringIO(text))

"""Strict CSV ingestion for long-format MOMENT inputs.

The boundary inspects the raw CSV header before pandas can normalize or mangle
ambiguous column names. Parsed frames still flow through ``validate_long_frame``
for the production validation contract.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

import pandas as pd

from .validation import ValidationError

CsvSource = bytes | bytearray | str | Path


def _validate_raw_header(header: list[str] | None) -> None:
    if header is None:
        raise ValidationError("EMPTY_CSV", "CSV input has no header row")

    normalized = [name.strip() for name in header]
    blank_positions = [index for index, name in enumerate(normalized) if not name]
    counts = Counter(normalized)
    duplicates = sorted(name for name, count in counts.items() if name and count > 1)

    if duplicates:
        raise ValidationError(
            "DUPLICATE_COLUMNS",
            "CSV header contains duplicate or whitespace-ambiguous column names",
            {"duplicates": duplicates},
        )
    if blank_positions:
        raise ValidationError(
            "AMBIGUOUS_COLUMNS",
            "CSV header contains blank column names",
            {"column_positions": blank_positions},
        )


def _header_from_text(text: str) -> list[str] | None:
    try:
        return next(csv.reader(io.StringIO(text, newline="")), None)
    except csv.Error as exc:
        raise ValidationError("INVALID_CSV_HEADER", f"CSV header could not be parsed: {exc}") from exc


def read_long_csv(source: CsvSource) -> pd.DataFrame:
    """Parse a long-format CSV only after validating its raw header.

    Exact and whitespace-equivalent duplicate names are rejected before
    ``pandas.read_csv`` can silently disambiguate them (for example ``value``
    becoming ``value.1``). Blank header identifiers are rejected for the same
    reason. This function parses only; callers must still invoke
    ``validate_long_frame`` to apply the full data contract.
    """

    if isinstance(source, (bytes, bytearray)):
        payload = bytes(source)
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValidationError(
                "INVALID_CSV_ENCODING",
                "CSV input must be UTF-8 encoded",
            ) from exc
        _validate_raw_header(_header_from_text(text))
        return pd.read_csv(io.BytesIO(payload))

    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(
            "INVALID_CSV_ENCODING",
            "CSV input must be UTF-8 encoded",
        ) from exc
    _validate_raw_header(_header_from_text(text))
    return pd.read_csv(path)

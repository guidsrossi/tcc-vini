from __future__ import annotations

import re
import unicodedata
import pandas as pd


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', normalize_text(text)).strip('-')


def normalize_text(text: str | None) -> str:
    value = '' if text is None else str(text)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    return value.lower().strip()


def human_int(value: int | float) -> str:
    return f'{int(value):,}'.replace(',', '.')


def pct(value: float, digits: int = 1) -> str:
    return f'{value * 100:.{digits}f}%'.replace('.', ',')


def risk_band(score: float) -> tuple[str, str]:
    if score < 0.20:
        return 'Baixa', '#ffb048'
    if score < 0.40:
        return 'Moderada', '#ff9c32'
    if score < 0.65:
        return 'Elevada', '#ff7d22'
    return 'Alta', '#ff6947'


def parse_number_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    raw = raw.replace({'nan': None, 'None': None, '': None})
    both = raw.str.contains(',', na=False) & raw.str.contains('.', na=False)
    comma_only = raw.str.contains(',', na=False) & ~raw.str.contains('.', na=False)
    cleaned = raw.copy()
    cleaned.loc[both] = cleaned.loc[both].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    cleaned.loc[comma_only] = cleaned.loc[comma_only].str.replace(',', '.', regex=False)
    return pd.to_numeric(cleaned, errors='coerce')

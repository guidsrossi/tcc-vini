from __future__ import annotations

import difflib
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from src.utils import normalize_text, risk_band

# ── Constants ───────────────────────────────────────────────────
WEEKDAYS = [
    'segunda-feira', 'terça-feira', 'quarta-feira',
    'quinta-feira',  'sexta-feira', 'sábado', 'domingo',
]

MIN_SUPPORT_RECORDS = 30
MIN_SUPPORT_HOURS   = 3

# Rush hours have higher baseline exposure — factored into time_score
RUSH_HOURS     = {7, 8, 17, 18, 19}
COMMUTE_HOURS  = {6, 9, 10, 16, 20}
NIGHT_HOURS    = {0, 1, 2, 3, 4, 5}

# Weather conditions that increase accident severity
ADVERSE_WEATHER = {
    'chuva', 'neblina', 'névoa', 'neblina/névoa', 'granizo', 'neve',
    'fumaca', 'fumaça', 'vento', 'tempestade',
}

# Behavioral / human-factor causes (increase responsibility weight)
BEHAVIORAL_CAUSE_KEYWORDS = [
    'atenção', 'alcool', 'álcool', 'velocidade', 'dormindo', 'sono',
    'celular', 'ultrapassagem', 'desobediência', 'imprudência',
    'distância', 'cansaço', 'fadiga',
]

# Single-lane or dangerous road types
RISKY_ROAD_TYPES = {
    'simples', 'mão única', 'mao unica', 'pista simples',
}

# Score component weights — sum to 1.0
W_FREQ     = 0.35   # frequency / volume
W_SEVERITY = 0.30   # injury/death severity
W_WEATHER  = 0.15   # adverse weather fraction
W_CAUSE    = 0.12   # behavioral cause fraction
W_ROAD     = 0.08   # risky road type fraction


# ── Low-level helpers ────────────────────────────────────────────
def _safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype='float')
    return pd.to_numeric(df[col], errors='coerce').fillna(default)


def _safe_text(df: pd.DataFrame, col: str, default: str = '') -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype='object')
    return df[col].fillna(default).astype(str)


def _normalize_log(series: pd.Series) -> pd.Series:
    """Log1p-normalize: compresses large counts, amplifies small ones."""
    s = pd.to_numeric(series, errors='coerce').fillna(0).clip(lower=0)
    s = np.log1p(s)
    max_val = float(s.max())
    if max_val <= 0:
        return pd.Series(0.0, index=series.index, dtype='float')
    return (s / max_val).clip(0, 1)


def _normalize_component(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(0)
    max_val = float(s.max()) if len(s) else 0.0
    if max_val <= 0:
        return pd.Series(0.0, index=series.index, dtype='float')
    return (s / max_val).clip(0, 1)


def _blend_with_support(raw_score: pd.Series, support_ratio: pd.Series, fallback: float) -> pd.Series:
    raw     = pd.to_numeric(raw_score, errors='coerce').fillna(fallback)
    support = pd.to_numeric(support_ratio, errors='coerce').fillna(0).clip(0, 1)
    return (raw * support + fallback * (1 - support)).clip(0, 1)


def _support_level(values: pd.Series, high_cut: float, mid_cut: float) -> pd.Series:
    values = pd.to_numeric(values, errors='coerce').fillna(0)
    return pd.Series(
        np.select([values >= high_cut, values >= mid_cut], ['Alta', 'Moderada'], default='Baixa'),
        index=values.index, dtype='object',
    )


def _confidence_label(records: int, observed_hours: int) -> str:
    if records >= 300 and observed_hours >= 10:
        return 'alta'
    if records >= 120 and observed_hours >= 6:
        return 'moderada'
    if records >= MIN_SUPPORT_RECORDS and observed_hours >= MIN_SUPPORT_HOURS:
        return 'baixa'
    return 'insuficiente'


def _is_adverse(weather_series: pd.Series) -> pd.Series:
    return weather_series.str.strip().str.lower().isin(ADVERSE_WEATHER).astype(float)


def _is_behavioral(cause_series: pd.Series) -> pd.Series:
    pattern = '|'.join(BEHAVIORAL_CAUSE_KEYWORDS)
    return cause_series.str.lower().str.contains(pattern, na=False).astype(float)


def _is_risky_road(road_series: pd.Series) -> pd.Series:
    return road_series.str.strip().str.lower().isin(RISKY_ROAD_TYPES).astype(float)


def _time_score_for_hour(hour: int) -> float:
    """Rush hours expose more traffic → higher baseline score."""
    if hour in RUSH_HOURS:
        return 1.0
    if hour in COMMUTE_HOURS:
        return 0.65
    if hour in NIGHT_HOURS:
        return 0.55   # night: fewer cars but higher severity
    return 0.35


def _top_value(series: pd.Series, n: int = 1) -> list[str]:
    counts = series.dropna().astype(str).value_counts()
    return counts.head(n).index.tolist()


# ── Feature engineering ──────────────────────────────────────────
def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    data  = df.copy()
    leves  = _safe_numeric(data, 'feridos_leves')
    graves = _safe_numeric(data, 'feridos_graves')
    mortos = _safe_numeric(data, 'mortos')

    # Calibrated, log-dampened severity weight
    data['risk_weight'] = (
        1.0
        + np.log1p(leves  * 0.40)
        + np.log1p(graves * 1.80)
        + np.log1p(mortos * 3.50)
    )

    data['month'] = pd.to_datetime(data.get('data_inversa'), errors='coerce').dt.month

    dias = _safe_text(data, 'dia_semana')
    data['is_weekend'] = dias.isin(['sábado', 'domingo']).astype(int)

    horas = _safe_numeric(data, 'hora', default=-1)
    data['periodo_dia'] = pd.cut(
        horas, bins=[-1, 5, 11, 17, 23],
        labels=['madrugada', 'manhã', 'tarde', 'noite'],
        include_lowest=True,
    ).astype(str)

    data['risk_component_severity'] = graves * 0.65 + mortos * 1.35 + leves * 0.15

    # New behavioral/contextual flags used in analysis
    data['is_adverse_weather'] = _is_adverse(_safe_text(data, 'condicao_metereologica'))
    data['is_behavioral_cause'] = _is_behavioral(_safe_text(data, 'causa_acidente'))
    data['is_risky_road'] = _is_risky_road(_safe_text(data, 'tipo_pista'))

    return data


# ── Filtering & UFs ─────────────────────────────────────────────
def filter_dataset(df: pd.DataFrame, source: str, years: list[int], ufs: list[str]) -> pd.DataFrame:
    out = df.copy()
    if source != 'Todas' and 'fonte_base' in out.columns:
        out = out[out['fonte_base'] == source]
    if years and 'ano_referencia' in out.columns:
        out = out[out['ano_referencia'].isin(years)]
    if ufs and 'uf' in out.columns:
        out = out[out['uf'].isin(ufs)]
    return out.reset_index(drop=True)


def available_ufs(df: pd.DataFrame, source: str) -> list[str]:
    base = filter_dataset(df, source, [], []) if source != 'Todas' else df
    return sorted(base['uf'].dropna().astype(str).unique().tolist()) if 'uf' in base.columns else []


# ── Overview ─────────────────────────────────────────────────────
def overview_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {'accidents': 0, 'serious_rate': 0.0, 'deaths': 0, 'grave_injuries': 0, 'score': 0.0}
    graves    = _safe_numeric(df, 'acidente_grave')
    deaths    = int(_safe_numeric(df, 'mortos').sum())
    grave_inj = int(_safe_numeric(df, 'feridos_graves').sum())
    rate      = float(graves.mean())
    adverse   = float(_safe_numeric(df, 'is_adverse_weather').mean())
    behavioral = float(_safe_numeric(df, 'is_behavioral_cause').mean())
    score = min(100.0, 100 * (
        rate   * 0.45
        + min((deaths + grave_inj) / max(len(df), 1), 1.0) * 0.30
        + adverse    * 0.15
        + behavioral * 0.10
    ))
    return {'accidents': int(len(df)), 'serious_rate': rate, 'deaths': deaths, 'grave_injuries': grave_inj, 'score': score}


# ── Main driver label ─────────────────────────────────────────── 
def _component_reason(row: pd.Series) -> str:
    components = [
        ('alto volume histórico',          float(row.get('freq_score', 0))),
        ('gravidade dos acidentes',         float(row.get('severity_score', 0))),
        ('condições climáticas adversas',   float(row.get('weather_score', 0))),
        ('causas comportamentais',          float(row.get('cause_score', 0))),
        ('tipo de pista mais perigoso',     float(row.get('road_score', 0))),
    ]
    components.sort(key=lambda x: x[1], reverse=True)
    return components[0][0]


# ── Find best consecutive window of hours ────────────────────────
def _best_consecutive_window(scores: dict[int, float], window: int = 2) -> list[int]:
    """Returns the consecutive window of `window` hours with lowest average score."""
    if not scores:
        return []
    hours = sorted(scores.keys())
    best_avg = float('inf')
    best_start = hours[0]
    for i in range(len(hours) - window + 1):
        block = hours[i:i + window]
        if block[-1] - block[0] == window - 1:   # consecutive
            avg = sum(scores[h] for h in block) / window
            if avg < best_avg:
                best_avg = avg
                best_start = block[0]
    return list(range(best_start, best_start + window))


# ── Hourly reference ─────────────────────────────────────────────
def hourly_reference(df: pd.DataFrame, day_name: str | None = None) -> pd.DataFrame:
    data = df.copy()
    if day_name:
        data = data[_safe_text(data, 'dia_semana') == day_name]

    all_hours = pd.DataFrame({'hora': list(range(24))})
    empty_cols = {
        'acidentes': 0, 'risco_medio': 0.0, 'taxa_grave': 0.0, 'observed': False,
        'support_ratio': 0.0, 'support_level': 'Sem dados',
        'freq_score': 0.0, 'severity_score': 0.0,
        'weather_score': 0.0, 'cause_score': 0.0, 'road_score': 0.0,
        'score_cautela': np.nan, 'score_100': np.nan,
        'faixa': 'Sem dados', 'driver': 'sem dados suficientes',
    }
    if data.empty:
        out = all_hours.copy()
        for col, val in empty_cols.items():
            out[col] = val
        return out

    grouped = data.groupby('hora', dropna=True).agg(
        acidentes         = ('uid_acidente',      'count'),
        risco_medio       = ('risk_weight',        'mean'),
        taxa_grave        = ('acidente_grave',     'mean'),
        adverse_frac      = ('is_adverse_weather', 'mean'),
        behavioral_frac   = ('is_behavioral_cause','mean'),
        risky_road_frac   = ('is_risky_road',      'mean'),
    ).reset_index()

    out = all_hours.merge(grouped, on='hora', how='left')
    for col, val in [('acidentes', 0), ('risco_medio', 0.0), ('taxa_grave', 0.0),
                     ('adverse_frac', 0.0), ('behavioral_frac', 0.0), ('risky_road_frac', 0.0)]:
        out[col] = out[col].fillna(val)
    out['acidentes']  = out['acidentes'].astype(int)
    out['observed']   = out['acidentes'] > 0

    median_acc = float(max(6, grouped['acidentes'].median())) if not grouped.empty else 6.0
    out['support_ratio']  = np.where(out['observed'], out['acidentes'] / (out['acidentes'] + median_acc), 0.0)
    out['support_level']  = _support_level(out['acidentes'], high_cut=max(16, median_acc * 1.5), mid_cut=max(6, median_acc))
    out.loc[~out['observed'], 'support_level'] = 'Sem dados'

    # Multi-factor score
    out['freq_score']     = _normalize_log(out['acidentes'])
    sev_base              = out['taxa_grave'] * 0.60 + _normalize_component(out['risco_medio']) * 0.40
    out['severity_score'] = _normalize_component(sev_base)
    out['weather_score']  = out['adverse_frac'].clip(0, 1)
    out['cause_score']    = out['behavioral_frac'].clip(0, 1)
    out['road_score']     = out['risky_road_frac'].clip(0, 1)
    out['time_factor']    = out['hora'].map(_time_score_for_hour)

    raw_score = (
        out['freq_score']     * W_FREQ
        + out['severity_score'] * W_SEVERITY
        + out['weather_score']  * W_WEATHER
        + out['cause_score']    * W_CAUSE
        + out['road_score']     * W_ROAD
    ).clip(0, 1)

    # Blend with time_factor as soft modifier (+10% max on nights/rush)
    raw_score = (raw_score * 0.92 + out['time_factor'] * raw_score * 0.08).clip(0, 1)

    fallback_score = float(raw_score[out['observed']].mean()) if out['observed'].any() else 0.0
    out['score_cautela'] = np.where(
        out['observed'],
        _blend_with_support(raw_score, out['support_ratio'], fallback_score),
        np.nan,
    )
    out['score_100'] = pd.Series(
        np.where(out['observed'], (pd.to_numeric(out['score_cautela']) * 100).round(), np.nan),
        index=out.index,
    ).astype('Int64')
    out['faixa']  = np.where(out['observed'], out['score_cautela'].map(lambda x: risk_band(float(x))[0]), 'Sem dados')
    out['driver'] = np.where(
        ~out['observed'], 'sem dados suficientes',
        out.apply(_component_reason, axis=1),
    )
    return out.sort_values('hora').reset_index(drop=True)


# ── Best / worst hours ───────────────────────────────────────────
def best_and_worst_hours(df: pd.DataFrame, day_name: str | None = None) -> dict:
    ref    = hourly_reference(df, day_name)
    usable = ref[ref['observed']].copy()
    if usable.empty:
        return {'best_window': [], 'worst_hours': [], 'best_hour': None, 'worst_hour': None}
    # Uma única hora observada não permite comparar horários. Marcá-la como a
    # melhor e a pior ao mesmo tempo produz uma recomendação contraditória.
    if len(usable) < 2:
        return {'best_window': [], 'worst_hours': [], 'best_hour': None, 'worst_hour': None}

    usable_sorted_asc  = usable.sort_values(['score_cautela', 'support_ratio', 'hora'], ascending=[True, False, True])
    usable_sorted_desc = usable.sort_values(['score_cautela', 'support_ratio', 'hora'], ascending=[False, False, True])

    best_raw   = usable_sorted_asc.head(6)['hora'].astype(int).tolist()
    # Try to find a consecutive window of 2 best hours
    score_map  = dict(zip(usable['hora'].astype(int), usable['score_cautela']))
    best_window = _best_consecutive_window(score_map, window=2) if len(usable) >= 3 else None
    best_window = best_window or best_raw[:1]
    best_set = set(best_window)
    worst_raw = [
        int(hour) for hour in usable_sorted_desc['hora'].tolist()
        if int(hour) not in best_set
    ][:3]

    return {
        'best_window': best_window,
        'worst_hours': worst_raw,
        'best_hour':   best_window[0] if best_window else None,
        'worst_hour':  worst_raw[0]   if worst_raw   else None,
    }


# ── Location reference ───────────────────────────────────────────
def location_reference(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            'municipio', 'uf', 'acidentes', 'graves', 'risco_medio', 'mortos',
            'latitude', 'longitude', 'taxa_grave', 'score_cautela', 'score_100',
            'faixa', 'principal_driver', 'support_level',
        ])
    grouped = df.groupby(['municipio', 'uf'], dropna=False).agg(
        acidentes        = ('uid_acidente',       'count'),
        graves           = ('acidente_grave',     'sum'),
        risco_medio      = ('risk_weight',         'mean'),
        mortos           = ('mortos',             'sum'),
        latitude         = ('latitude',           'median'),
        longitude        = ('longitude',          'median'),
        adverse_frac     = ('is_adverse_weather', 'mean'),
        behavioral_frac  = ('is_behavioral_cause','mean'),
        risky_road_frac  = ('is_risky_road',      'mean'),
    ).reset_index()

    grouped['taxa_grave']     = grouped['graves'] / grouped['acidentes'].clip(lower=1)
    grouped['freq_score']     = _normalize_log(grouped['acidentes'])
    severity                  = (grouped['taxa_grave'].fillna(0) * 0.55
                                 + _normalize_component(grouped['risco_medio']) * 0.25
                                 + _normalize_component(grouped['mortos']) * 0.20)
    grouped['severity_score'] = _normalize_component(severity)
    grouped['weather_score']  = grouped['adverse_frac'].fillna(0).clip(0, 1)
    grouped['cause_score']    = grouped['behavioral_frac'].fillna(0).clip(0, 1)
    grouped['road_score']     = grouped['risky_road_frac'].fillna(0).clip(0, 1)
    grouped['time_score']     = 0.0   # location has no single time, skip

    raw_score = (
        grouped['freq_score']     * W_FREQ
        + grouped['severity_score'] * W_SEVERITY
        + grouped['weather_score']  * W_WEATHER
        + grouped['cause_score']    * W_CAUSE
        + grouped['road_score']     * W_ROAD
    ).clip(0, 1)

    grouped['support_ratio']  = grouped['acidentes'] / (grouped['acidentes'] + 30)
    grouped['score_cautela']  = _blend_with_support(raw_score, grouped['support_ratio'], float(raw_score.mean()) if len(raw_score) else 0.0)
    grouped['score_100']      = (grouped['score_cautela'] * 100).round().astype(int)
    grouped['faixa']          = grouped['score_cautela'].map(lambda x: risk_band(float(x))[0])
    grouped['principal_driver'] = grouped.apply(_component_reason, axis=1)
    grouped['support_level']  = _support_level(grouped['acidentes'], high_cut=60, mid_cut=20)
    return grouped.sort_values(['score_cautela', 'acidentes'], ascending=[False, False]).reset_index(drop=True)


# ── Road reference ────────────────────────────────────────────────
def road_reference(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            'rodovia', 'acidentes', 'graves', 'mortos', 'taxa_grave',
            'score_cautela', 'score_100', 'faixa', 'principal_driver', 'support_level',
        ])
    grouped = df.groupby('rodovia', dropna=False).agg(
        acidentes       = ('uid_acidente',       'count'),
        graves          = ('acidente_grave',     'sum'),
        mortos          = ('mortos',             'sum'),
        risco_medio     = ('risk_weight',         'mean'),
        adverse_frac    = ('is_adverse_weather', 'mean'),
        behavioral_frac = ('is_behavioral_cause','mean'),
        risky_road_frac = ('is_risky_road',      'mean'),
    ).reset_index()

    grouped['taxa_grave']     = grouped['graves'] / grouped['acidentes'].clip(lower=1)
    grouped['freq_score']     = _normalize_log(grouped['acidentes'])
    severity                  = (grouped['taxa_grave'].fillna(0) * 0.65
                                 + _normalize_component(grouped['mortos']) * 0.25
                                 + _normalize_component(grouped['risco_medio']) * 0.10)
    grouped['severity_score'] = _normalize_component(severity)
    grouped['weather_score']  = grouped['adverse_frac'].fillna(0).clip(0, 1)
    grouped['cause_score']    = grouped['behavioral_frac'].fillna(0).clip(0, 1)
    grouped['road_score']     = grouped['risky_road_frac'].fillna(0).clip(0, 1)
    grouped['time_score']     = 0.0

    raw_score = (
        grouped['freq_score']     * W_FREQ
        + grouped['severity_score'] * W_SEVERITY
        + grouped['weather_score']  * W_WEATHER
        + grouped['cause_score']    * W_CAUSE
        + grouped['road_score']     * W_ROAD
    ).clip(0, 1)

    grouped['support_ratio']  = grouped['acidentes'] / (grouped['acidentes'] + 45)
    grouped['score_cautela']  = _blend_with_support(raw_score, grouped['support_ratio'], float(raw_score.mean()) if len(raw_score) else 0.0)
    grouped['score_100']      = (grouped['score_cautela'] * 100).round().astype(int)
    grouped['faixa']          = grouped['score_cautela'].map(lambda x: risk_band(float(x))[0])
    grouped['principal_driver'] = grouped.apply(_component_reason, axis=1)
    grouped['support_level']  = _support_level(grouped['acidentes'], high_cut=90, mid_cut=30)
    return grouped.sort_values(['score_cautela', 'acidentes'], ascending=[False, False]).reset_index(drop=True)


# ── Destination search ────────────────────────────────────────────
def _destination_labels(df: pd.DataFrame) -> list[str]:
    if 'municipio' not in df.columns:
        return []
    return sorted(df['municipio'].dropna().astype(str).unique().tolist())


def suggest_destinations(df: pd.DataFrame, destination: str, limit: int = 6) -> list[str]:
    norm_target = normalize_text(destination)
    labels      = _destination_labels(df)
    if not norm_target or not labels:
        return []
    mapping = {normalize_text(x): x for x in labels}
    starts   = [orig for norm, orig in mapping.items() if norm.startswith(norm_target)]
    contains = [orig for norm, orig in mapping.items() if norm_target in norm and orig not in starts]
    fuzzy    = [mapping[m] for m in difflib.get_close_matches(norm_target, list(mapping.keys()), n=limit, cutoff=0.68)]
    ordered: list[str] = []
    for bucket in [starts, contains, fuzzy]:
        for item in bucket:
            if item not in ordered:
                ordered.append(item)
    return ordered[:limit]


def resolve_destination(df: pd.DataFrame, destination: str) -> tuple[str | None, pd.DataFrame]:
    norm_target = normalize_text(destination)
    if not norm_target or 'municipio' not in df.columns:
        return None, df.iloc[0:0].copy()
    labels  = _destination_labels(df)
    mapping = {normalize_text(x): x for x in labels}
    if norm_target in mapping:
        label = mapping[norm_target]
        return label, df[df['municipio'] == label].copy()
    contains = [orig for norm, orig in mapping.items() if norm_target in norm or norm.startswith(norm_target)]
    if contains:
        label = sorted(contains, key=lambda x: (len(x), x))[0]
        return label, df[df['municipio'] == label].copy()
    matches = difflib.get_close_matches(norm_target, list(mapping.keys()), n=1, cutoff=0.72)
    if matches:
        label = mapping[matches[0]]
        return label, df[df['municipio'] == label].copy()
    return None, df.iloc[0:0].copy()


def build_maps_link(origin: str, destination: str) -> str:
    return (
        f'https://www.google.com/maps/dir/?api=1'
        f'&origin={quote_plus(origin)}'
        f'&destination={quote_plus(destination)}'
        f'&travelmode=driving'
    )


# ── Destination snapshot ──────────────────────────────────────────
def destination_snapshot(df: pd.DataFrame, day_name: str) -> dict[str, object]:
    ref          = hourly_reference(df, day_name)
    observed_ref = ref[ref['observed']].copy()
    picks        = best_and_worst_hours(df, day_name)

    empty_table = pd.DataFrame(columns=['hora', 'acidentes', 'score_100', 'faixa', 'driver', 'support_level'])

    if df.empty or observed_ref.empty:
        return {
            'score': 0.0, 'score_100': 0, 'band': 'Sem dados',
            'confidence': 'insuficiente', 'best_hours': [], 'worst_hours': [],
            'best_table': empty_table.copy(), 'worst_table': empty_table.copy(),
            'reasons': ['não há dados suficientes para indicar um horário com segurança'],
            'serious_rate': 0.0, 'records': int(len(df)),
            'observed_hours': 0, 'volatility': 0.0,
            'top_cause': None, 'adverse_rate': 0.0, 'behavioral_rate': 0.0,
        }

    # Volume-weighted score — hours with more accidents carry more weight
    weights = observed_ref['acidentes'].values.astype(float)
    weights = np.where(weights == 0, 1.0, weights)
    score   = float(np.average(observed_ref['score_cautela'].values, weights=weights))

    band, _       = risk_band(score)
    observed_hours = int(observed_ref['hora'].nunique())
    confidence    = _confidence_label(int(len(df)), observed_hours)

    # Volatility: std of hourly scores → how predictable the pattern is
    volatility = float(observed_ref['score_cautela'].std()) if len(observed_ref) > 1 else 0.0

    # Context rates for richer narrative
    adverse_rate    = float(_safe_numeric(df, 'is_adverse_weather').mean())
    behavioral_rate = float(_safe_numeric(df, 'is_behavioral_cause').mean())
    top_causes      = _top_value(_safe_text(df, 'causa_acidente').replace({'': None, 'nan': None}).dropna(), n=2)
    top_cause       = top_causes[0] if top_causes else None
    if top_cause:
        top_cause = str(top_cause).replace('Rea��o', 'Reação').replace('rea��o', 'reação')

    # Build reasons
    reasons: list[str] = []
    worst_row = observed_ref.sort_values(['score_cautela', 'acidentes'], ascending=[False, False]).head(1)
    if not worst_row.empty:
        wr = worst_row.iloc[0]
        reasons.append(f"o horário mais crítico histórico fica perto de {int(wr['hora']):02d}h ({wr['driver']})")
    if top_cause:
        reasons.append(f"causa mais comum nos dados: {top_cause.lower()}")
    if adverse_rate > 0.10:
        reasons.append(f"{adverse_rate * 100:.0f}% dos casos ocorreram em condições climáticas adversas")
    if behavioral_rate > 0.20:
        reasons.append(f"{behavioral_rate * 100:.0f}% dos casos têm causa comportamental registrada")
    if volatility > 0.15:
        reasons.append('o padrão de risco varia bastante ao longo do dia — escolher bem a hora faz diferença')
    elif len(observed_ref) >= 4:
        reasons.append('o padrão de risco é relativamente estável — o horário tem menos impacto')
    if confidence == 'insuficiente':
        reasons.append('a quantidade de dados ainda é pequena para uma resposta mais firme')

    cols_table = ['hora', 'acidentes', 'support_level', 'score_100', 'faixa', 'driver']
    best_table  = observed_ref.sort_values(['score_cautela', 'support_ratio', 'hora'], ascending=[True, False, True]).head(6)[cols_table].copy()
    worst_table = observed_ref.sort_values(['score_cautela', 'support_ratio', 'hora'], ascending=[False, False, True]).head(6)[cols_table].copy()

    return {
        'score': score, 'score_100': int(round(score * 100)), 'band': band,
        'confidence': confidence, 'best_hours': picks['best_window'], 'worst_hours': picks['worst_hours'],
        'best_table': best_table, 'worst_table': worst_table,
        'reasons': reasons[:5],
        'serious_rate': float(_safe_numeric(df, 'acidente_grave').mean()),
        'records': int(len(df)), 'observed_hours': observed_hours,
        'volatility': volatility, 'top_cause': top_cause,
        'adverse_rate': adverse_rate, 'behavioral_rate': behavioral_rate,
    }


# ── Explain trip ──────────────────────────────────────────────────
def explain_trip(df: pd.DataFrame, day_name: str, destination_label: str | None) -> dict[str, object]:
    snap     = destination_snapshot(df, day_name)
    dest_txt = destination_label or 'informado'

    best_txt  = ', '.join([f'{h:02d}h' for h in snap['best_hours'][:2]])  if snap['best_hours']  else 'sem referência suficiente'
    worst_txt = ', '.join([f'{h:02d}h' for h in snap['worst_hours']])     if snap['worst_hours'] else 'sem destaque claro'

    # Contextual action based on band + behavioral rate
    if snap['confidence'] == 'insuficiente':
        action = 'Há poucos dados para uma resposta firme. Use como pista e confirme a rota no Maps.'
    elif snap['band'] in ('Alta', 'Elevada') and snap['behavioral_rate'] > 0.25:
        action = 'Aqui vale ter muito cuidado. Evite os piores horários, dirija com atenção redobrada e verifique condições da via antes de sair.'
    elif snap['band'] in ('Alta', 'Elevada'):
        action = 'Este trecho pede atenção. Se puder, prefira os melhores horários e use o GPS atualizado.'
    else:
        action = 'Se puder escolher, use o horário de melhor avaliação e confirme a rota no Maps antes de sair.'

    # Volatility note
    vol_note = ''
    if snap['volatility'] > 0.15:
        vol_note = ' O padrão varia bastante ao longo do dia — escolher o horário certo faz diferença real.'
    elif snap['volatility'] < 0.06 and snap['observed_hours'] >= 6:
        vol_note = ' O padrão é estável ao longo do dia — o horário tem pouco impacto nesse destino.'

    cause_note = f' Causa mais registrada: {snap["top_cause"].lower()}.' if snap['top_cause'] else ''
    weather_note = f' {snap["adverse_rate"]*100:.0f}% dos casos em clima adverso.' if snap['adverse_rate'] > 0.10 else ''

    summary = (
        f'Para {dest_txt} em {day_name}, a nota de atenção é {snap["score_100"]}/100 '
        f'(nível {str(snap["band"]).lower()}). '
        f'Melhor horário: {best_txt}. '
        f'Evitar: {worst_txt}.{vol_note}{cause_note}{weather_note}'
    )
    recommendation = (
        f'Confiança: {snap["confidence"]} · {snap["records"]} registros · '
        f'{snap["observed_hours"]} horas com dados · '
        f'{snap["serious_rate"]*100:.1f}% casos graves. {action}'
    )

    return {
        'summary': summary, 'recommendation': recommendation,
        'best_hours': snap['best_hours'], 'worst_hours': snap['worst_hours'],
        'score_100': snap['score_100'], 'band': snap['band'],
        'confidence': snap['confidence'], 'reasons': snap['reasons'],
        'best_table': snap['best_table'], 'worst_table': snap['worst_table'],
        'records': snap['records'], 'observed_hours': snap['observed_hours'],
        'serious_rate': snap['serious_rate'],
    }


# ── Aggregates used by dashboard pages ───────────────────────────
def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'ano_referencia' not in df.columns:
        return pd.DataFrame(columns=['ano', 'acidentes', 'mortos', 'graves'])
    grouped = (
        df.groupby('ano_referencia', dropna=True)
        .agg(acidentes=('uid_acidente', 'count'), mortos=('mortos', 'sum'), graves=('feridos_graves', 'sum'))
        .reset_index()
        .rename(columns={'ano_referencia': 'ano'})
    )
    grouped['ano'] = grouped['ano'].astype(str)
    return grouped.sort_values('ano').reset_index(drop=True)


def hourly_day_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'hora' not in df.columns or 'dia_semana' not in df.columns:
        return pd.DataFrame()
    data = df.copy()
    data['hora']      = pd.to_numeric(data['hora'], errors='coerce')
    data['dia_semana'] = data['dia_semana'].astype(str)
    data = data[data['dia_semana'].isin(WEEKDAYS) & data['hora'].notna()]
    if data.empty:
        return pd.DataFrame()
    grouped = data.groupby(['dia_semana', 'hora']).agg(
        acidentes       = ('uid_acidente',       'count'),
        graves          = ('acidente_grave',     'mean'),
        adverse_frac    = ('is_adverse_weather', 'mean'),
        behavioral_frac = ('is_behavioral_cause','mean'),
    ).reset_index()
    max_acc = float(grouped['acidentes'].max()) or 1.0
    # Multi-factor score in heatmap too
    grouped['score'] = (
        (np.log1p(grouped['acidentes']) / np.log1p(max_acc)) * 0.45
        + grouped['graves'].fillna(0)                         * 0.30
        + grouped['adverse_frac'].fillna(0)                   * 0.15
        + grouped['behavioral_frac'].fillna(0)                * 0.10
    ) * 100
    pivot = grouped.pivot(index='dia_semana', columns='hora', values='score')
    pivot = pivot.reindex(WEEKDAYS)
    pivot.columns = [int(c) for c in pivot.columns]
    return pivot

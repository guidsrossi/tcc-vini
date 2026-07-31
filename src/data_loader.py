from __future__ import annotations

from pathlib import Path
import csv
import re
import pandas as pd

from src.config import DATA_DIR
from src.utils import parse_number_series

REQUIRED_CANONICAL_COLUMNS = {
    'uid_acidente', 'data_inversa', 'uf', 'municipio', 'hora', 'fonte_base',
    'feridos_leves', 'feridos_graves', 'mortos', 'acidente_grave', 'rodovia', 'ano_referencia'
}

MULTIPART_SUFFIX = re.compile(r'_parte_\d+$', flags=re.I)


def _logical_source_stem(path: Path) -> str:
    """Return the original dataset stem for a file split into GitHub-safe parts."""
    return MULTIPART_SUFFIX.sub('', path.stem)


def _logical_source_name(path: Path) -> str:
    return f'{_logical_source_stem(path)}{path.suffix}'


def _sniff_separator(path: Path, encoding: str) -> str:
    with path.open('r', encoding=encoding, errors='ignore') as fh:
        sample = fh.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=';,\t|')
        return dialect.delimiter
    except Exception:
        return ';' if sample.count(';') >= sample.count(',') else ','


def _read_csv(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ('utf-8', 'latin1'):
        try:
            sep = _sniff_separator(path, encoding)
            return pd.read_csv(path, encoding=encoding, sep=sep)
        except Exception as exc:
            errors.append(f'{encoding}: {exc}')
    raise ValueError(f'Não foi possível ler {path.name}. Tentativas: {errors}')


def _extract_year(path: Path) -> int | None:
    match = re.search(r'(20\d{2})', path.stem)
    return int(match.group(1)) if match else None


def _safe_first(df: pd.DataFrame, candidates: list[str], default=None):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return default


def _normalize_national(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ['km', 'latitude', 'longitude', 'idade']:
        if col in df.columns:
            df[col] = parse_number_series(df[col])
    for col in ['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0
    if 'data_inversa' in df.columns:
        df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')
    else:
        df['data_inversa'] = pd.NaT
    extracted_year = _extract_year(path)
    df['ano_referencia'] = df['data_inversa'].dt.year
    if extracted_year is not None:
        df['ano_referencia'] = df['ano_referencia'].fillna(extracted_year)
    df['ano_referencia'] = df['ano_referencia'].astype('Int64')
    if 'horario' in df.columns:
        hora = pd.to_datetime(df['horario'], format='%H:%M:%S', errors='coerce').dt.hour
        if hora.isna().all():
            hora = pd.to_datetime(df['horario'], errors='coerce').dt.hour
        df['hora'] = hora
    else:
        df['hora'] = pd.NA
    df['fonte_base'] = 'Base nacional'
    # Multipart files are normalized as one logical source.  Keeping the
    # original name and UID makes splitting transparent to the dashboard.
    source_stem = _logical_source_stem(path)
    df['arquivo_origem'] = _logical_source_name(path)
    if 'id' in df.columns:
        df['uid_acidente'] = source_stem + '_' + df['id'].astype(str)
    else:
        df['uid_acidente'] = source_stem + '_' + df.index.astype(str)
    first_cols = [
        'uid_acidente', 'data_inversa', 'dia_semana', 'uf', 'br', 'km', 'municipio', 'causa_principal', 'causa_acidente',
        'tipo_acidente', 'classificacao_acidente', 'fase_dia', 'condicao_metereologica', 'tipo_pista', 'tracado_via',
        'uso_solo', 'latitude', 'longitude', 'regional', 'delegacia', 'uop', 'ano_referencia', 'hora', 'fonte_base', 'arquivo_origem'
    ]
    agg = {c: 'first' for c in first_cols if c in df.columns}
    for col in ['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']:
        agg[col] = 'sum'
    out = df.groupby('uid_acidente', as_index=False).agg(agg)
    out['acidente_grave'] = ((out['mortos'] > 0) | (out['feridos_graves'] > 0)).astype(int)
    br_text = out.get('br', pd.Series(index=out.index, dtype='object')).fillna('')
    out['rodovia'] = br_text.map(lambda x: f'BR-{int(x)}' if str(x).strip().replace('.0', '').isdigit() else 'Não informada')
    return out


def _normalize_sp(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {
        '_id': 'id', 'marco_qm': 'km', 'data': 'data_inversa', 'hr_acid': 'horario', 'class_acid': 'classificacao_acidente',
        'tipo_acid': 'tipo_acidente', 'causa': 'causa_acidente', 'meteoro': 'condicao_metereologica',
        'qtd_vit_ilesa': 'ilesos', 'qtd_vit_fatal': 'mortos', 'qtd_vit_grave': 'feridos_graves', 'qtd_vit_leve': 'feridos_leves',
        'regional_der': 'regional', 'visib': 'fase_dia'
    }
    df = df.rename(columns=rename)
    for col in ['km', 'latitude', 'longitude']:
        if col in df.columns:
            df[col] = parse_number_series(df[col])
    df['data_inversa'] = pd.to_datetime(df.get('data_inversa'), errors='coerce')
    extracted_year = _extract_year(path)
    df['ano_referencia'] = df['data_inversa'].dt.year
    if extracted_year is not None:
        df['ano_referencia'] = df['ano_referencia'].fillna(extracted_year)
    df['ano_referencia'] = df['ano_referencia'].astype('Int64')
    if 'horario' in df.columns:
        hora = pd.to_datetime(df['horario'], format='%H:%M', errors='coerce').dt.hour
        if hora.isna().all():
            hora = pd.to_datetime(df['horario'], errors='coerce').dt.hour
        df['hora'] = hora
    else:
        df['hora'] = pd.NA
    weekdays = {
        'Monday': 'segunda-feira', 'Tuesday': 'terça-feira', 'Wednesday': 'quarta-feira', 'Thursday': 'quinta-feira',
        'Friday': 'sexta-feira', 'Saturday': 'sábado', 'Sunday': 'domingo'
    }
    day_name = df['data_inversa'].dt.day_name()
    df['dia_semana'] = day_name.map(weekdays).fillna(day_name.str.lower())
    df['uf'] = 'SP'
    rodovia_series = _safe_first(df, ['rodovia', 'sp'])
    if isinstance(rodovia_series, pd.Series):
        df['br'] = rodovia_series.astype(str).str.extract(r'(\d+)')
        df['rodovia'] = rodovia_series.fillna('Não informada').astype(str)
    else:
        df['br'] = pd.NA
        df['rodovia'] = 'Não informada'
    for col, default in {
        'tracado_via': 'Não informado', 'fase_dia': 'Não informado', 'uso_solo': 'Não informado',
        'causa_principal': 'Não informado', 'tipo_pista': 'Não informado', 'municipio': 'Não informado'
    }.items():
        if col not in df.columns:
            df[col] = default
    df['arquivo_origem'] = path.name
    df['fonte_base'] = 'Base SP'
    if 'id' in df.columns:
        df['uid_acidente'] = path.stem + '_' + df['id'].astype(str)
    else:
        df['uid_acidente'] = path.stem + '_' + df.index.astype(str)
    for col in ['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0
    df['acidente_grave'] = ((df['mortos'] > 0) | (df['feridos_graves'] > 0)).astype(int)
    cols = [
        'uid_acidente', 'data_inversa', 'dia_semana', 'uf', 'br', 'rodovia', 'km', 'municipio', 'causa_principal', 'causa_acidente',
        'tipo_acidente', 'classificacao_acidente', 'fase_dia', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo',
        'latitude', 'longitude', 'regional', 'ano_referencia', 'hora', 'fonte_base', 'arquivo_origem', 'ilesos', 'feridos_leves',
        'feridos_graves', 'mortos', 'acidente_grave'
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def discover_data_files() -> dict[str, list[Path]]:
    all_csv = sorted(DATA_DIR.glob('*.csv'))
    sp = [p for p in all_csv if re.search(r'sp.*der|der.*sp', p.stem, flags=re.I)]
    national = [p for p in all_csv if p not in sp and re.search(r'acidentes.*(202|20)', p.stem, flags=re.I)]
    return {'Base nacional': national, 'Base SP': sp}


def validate_loaded_data(df: pd.DataFrame) -> dict[str, object]:
    missing = sorted(REQUIRED_CANONICAL_COLUMNS - set(df.columns))
    source_counts = df['fonte_base'].value_counts(dropna=False).to_dict() if 'fonte_base' in df.columns else {}
    national_ufs = sorted(df.loc[df.get('fonte_base', pd.Series(dtype='object')) == 'Base nacional', 'uf'].dropna().astype(str).unique().tolist()) if 'uf' in df.columns else []
    return {
        'rows': int(len(df)),
        'missing_columns': missing,
        'source_counts': source_counts,
        'national_uf_count': len(national_ufs),
        'national_ufs': national_ufs,
        'ok': not missing and len(df) > 0,
    }


def load_accident_data() -> pd.DataFrame:
    groups = discover_data_files()
    frames: list[pd.DataFrame] = []
    for path in groups['Base nacional']:
        frames.append(_normalize_national(_read_csv(path), path))
    for path in groups['Base SP']:
        frames.append(_normalize_sp(_read_csv(path), path))
    if not frames:
        raise FileNotFoundError(f'Nenhuma base encontrada em {DATA_DIR}')
    df = pd.concat(frames, ignore_index=True, sort=False)
    df['municipio'] = df.get('municipio', pd.Series(index=df.index, dtype='object')).fillna('Não informado').astype(str).str.upper().str.strip()
    df['uf'] = df.get('uf', pd.Series(index=df.index, dtype='object')).fillna('NI').astype(str).str.upper().str.strip()
    df['rodovia'] = df.get('rodovia', pd.Series(index=df.index, dtype='object')).fillna('Não informada').astype(str).str.strip()
    df['hora'] = pd.to_numeric(df.get('hora', pd.Series(index=df.index, dtype='float')), errors='coerce').clip(lower=0, upper=23)
    df['ano_referencia'] = pd.to_numeric(df.get('ano_referencia', pd.Series(index=df.index, dtype='float')), errors='coerce').astype('Int64')
    validation = validate_loaded_data(df)
    if not validation['ok']:
        raise ValueError(f"Base carregada com problema de schema: {validation['missing_columns']}")
    return df

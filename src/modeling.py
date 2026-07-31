from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.config import MODELS_DIR, OUTPUTS_DIR

warnings.filterwarnings('ignore', category=UserWarning)

# ── Feature groups ─────────────────────────────────────────────
FEATURES = [
    'uf', 'hora', 'dia_semana', 'municipio', 'rodovia',
    'causa_acidente', 'tipo_acidente', 'condicao_metereologica',
    'tipo_pista', 'tracado_via', 'fonte_base',
    'month', 'is_weekend', 'periodo_dia',
]

# hora and month get cyclic encoding: sin + cos → preserves circularity
CYCLIC_MAP: dict[str, int] = {'hora': 24, 'month': 12}

# Passthrough as numeric
NUMERIC_PASS = ['is_weekend']

# Categorical → OHE
CAT_FEATURES = [
    'uf', 'dia_semana', 'municipio', 'rodovia',
    'causa_acidente', 'tipo_acidente', 'condicao_metereologica',
    'tipo_pista', 'tracado_via', 'fonte_base', 'periodo_dia',
]

MODEL_PATH = MODELS_DIR / 'modelo_risco.pkl'


# ── Cyclic encoding ─────────────────────────────────────────────
def _cyclic_encode(X: pd.DataFrame, periods: dict[str, int]) -> np.ndarray:
    """Encode cyclic features as (sin, cos) pairs. Midnight→0 and 23h stay adjacent."""
    cols = [c for c in periods if c in X.columns]
    parts: list[np.ndarray] = []
    for col in cols:
        vals = pd.to_numeric(X[col], errors='coerce').fillna(0).values
        period = periods[col]
        parts.append(np.sin(2 * np.pi * vals / period))
        parts.append(np.cos(2 * np.pi * vals / period))
    return np.column_stack(parts) if parts else np.zeros((len(X), 2))


# ── Data preparation ────────────────────────────────────────────
def prepare_training_frame(df: pd.DataFrame, max_rows: int = 100_000) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy().dropna(subset=['acidente_grave'])
    keep = [c for c in FEATURES if c in data.columns]
    base = data[keep + ['acidente_grave', 'ano_referencia']].copy()
    base['acidente_grave'] = (
        pd.to_numeric(base['acidente_grave'], errors='coerce').fillna(0).astype(int)
    )
    if base['acidente_grave'].nunique() < 2:
        raise ValueError('A base de modelagem não possui duas classes para treino.')

    if len(base) > max_rows:
        frac = max_rows / len(base)
        parts = []
        for _, grp in base.groupby('acidente_grave'):
            n = max(1, int(len(grp) * frac))
            parts.append(grp.sample(min(n, len(grp)), random_state=42))
        base = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)

    return base, keep


def split_frame(base: pd.DataFrame, feature_cols: list[str]) -> dict:
    years = sorted([int(x) for x in base['ano_referencia'].dropna().unique()])
    if len(years) >= 2:
        test_year   = years[-1]
        train_base  = base[base['ano_referencia'] < test_year]
        test_base   = base[base['ano_referencia'] == test_year]
        two_classes = lambda s: s['acidente_grave'].nunique() == 2
        if len(train_base) > 1000 and two_classes(train_base) and two_classes(test_base):
            X_tr, X_val, y_tr, y_val = train_test_split(
                train_base[feature_cols], train_base['acidente_grave'].astype(int),
                test_size=0.15, random_state=42, stratify=train_base['acidente_grave'],
            )
            return {
                'mode': 'holdout_temporal', 'train_years': years[:-1], 'test_year': test_year,
                'X_train': X_tr, 'y_train': y_tr, 'X_val': X_val, 'y_val': y_val,
                'X_test': test_base[feature_cols], 'y_test': test_base['acidente_grave'].astype(int),
            }
    X      = base[feature_cols]
    y      = base['acidente_grave'].astype(int)
    X_all, X_test, y_all, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr,  X_val,  y_tr,  y_val  = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
    return {
        'mode': 'split_estratificado', 'train_years': [], 'test_year': None,
        'X_train': X_tr, 'y_train': y_tr, 'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


# ── Threshold via precision-recall curve ────────────────────────
def choose_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Choose threshold that maximizes F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = np.where(
        (precision + recall) == 0, 0.0,
        2 * precision * recall / (precision + recall),
    )
    best_idx = int(np.argmax(f1[:-1]))
    return float(np.clip(thresholds[best_idx], 0.25, 0.75))


# ── Pipeline ────────────────────────────────────────────────────
def build_pipeline(feature_cols: list[str]) -> Pipeline:
    cyclic_cols  = [c for c in feature_cols if c in CYCLIC_MAP]
    numeric_cols = [c for c in feature_cols if c in NUMERIC_PASS]
    cat_cols     = [c for c in feature_cols if c in CAT_FEATURES]

    transformers: list = []

    if cyclic_cols:
        periods = {c: CYCLIC_MAP[c] for c in cyclic_cols}
        transformers.append((
            'cyclic',
            FunctionTransformer(lambda X, p=periods: _cyclic_encode(X, p), validate=False),
            cyclic_cols,
        ))

    if numeric_cols:
        transformers.append(('num', SimpleImputer(strategy='median'), numeric_cols))

    if cat_cols:
        transformers.append((
            'cat',
            Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('enc', OneHotEncoder(handle_unknown='ignore', min_frequency=8, sparse_output=False)),
            ]),
            cat_cols,
        ))

    pre = ColumnTransformer(transformers, remainder='drop')

    # HistGradientBoostingClassifier:
    #   - handles missing values natively after preprocessing
    #   - early stopping avoids overfitting without manual depth tuning
    #   - consistently outperforms RandomForest on tabular data
    clf = HistGradientBoostingClassifier(
        max_iter=600,
        max_depth=8,
        min_samples_leaf=25,
        learning_rate=0.05,
        l2_regularization=0.5,
        class_weight='balanced',
        random_state=42,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=25,
        verbose=0,
    )

    return Pipeline([('pre', pre), ('clf', clf)])


# ── Cross-validation ────────────────────────────────────────────
def cross_validate_pipeline(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    roc = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    f1  = cross_val_score(pipe, X, y, cv=skf, scoring='f1',      n_jobs=-1)
    return {
        'cv_roc_auc_mean': float(roc.mean()),
        'cv_roc_auc_std':  float(roc.std()),
        'cv_f1_mean':      float(f1.mean()),
        'cv_f1_std':       float(f1.std()),
    }


# ── Feature importances ─────────────────────────────────────────
def get_feature_importances(pipe: Pipeline, top_n: int = 20) -> list[dict]:
    try:
        clf  = pipe.named_steps['clf']
        pre  = pipe.named_steps['pre']
        if not hasattr(clf, 'feature_importances_'):
            return []
        names = pre.get_feature_names_out()
        fi    = clf.feature_importances_
        pairs = sorted(zip(names.tolist(), fi.tolist()), key=lambda x: x[1], reverse=True)
        return [{'feature': n, 'importance': round(float(v), 6)} for n, v in pairs[:top_n]]
    except Exception:
        return []


# ── Train & evaluate ────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame) -> dict:
    base, features = prepare_training_frame(df)
    split          = split_frame(base, features)

    # Cross-validate on full training pool (train + val)
    X_pool = pd.concat([split['X_train'], split['X_val']], ignore_index=True)
    y_pool = pd.concat([split['y_train'], split['y_val']], ignore_index=True)
    cv_metrics = cross_validate_pipeline(build_pipeline(features), X_pool, y_pool, cv=5)

    # Final model: train on pool, threshold tuned on val, evaluate on test
    pipe = build_pipeline(features)
    pipe.fit(X_pool, y_pool)

    threshold  = choose_threshold(split['y_val'], pipe.predict_proba(split['X_val'])[:, 1])
    test_proba = pipe.predict_proba(split['X_test'])[:, 1]
    pred       = (test_proba >= threshold).astype(int)

    try:
        roc_auc = float(roc_auc_score(split['y_test'], test_proba))
    except Exception:
        roc_auc = 0.0

    metrics: dict = {
        'accuracy':              float(accuracy_score(split['y_test'], pred)),
        'precision':             float(precision_score(split['y_test'], pred, zero_division=0)),
        'recall':                float(recall_score(split['y_test'], pred, zero_division=0)),
        'f1_score':              float(f1_score(split['y_test'], pred, zero_division=0)),
        'roc_auc':               roc_auc,
        'threshold':             float(threshold),
        'samples':               int(len(base)),
        'positive_rate':         float(base['acidente_grave'].mean()),
        'confusion_matrix':      confusion_matrix(split['y_test'], pred).tolist(),
        'classification_report': classification_report(split['y_test'], pred, zero_division=0),
        'mode':                  split['mode'],
        'train_years':           split['train_years'],
        'test_year':             split['test_year'],
        'features':              features,
        'feature_importances':   get_feature_importances(pipe),
        **cv_metrics,
    }
    return {'pipeline': pipe, 'threshold': threshold, 'metrics': metrics}


# ── Persist artifacts ───────────────────────────────────────────
def save_artifacts(artifact: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    safe_metrics = {k: v for k, v in artifact['metrics'].items() if k != 'classification_report'}
    (OUTPUTS_DIR / 'model_metrics.json').write_text(
        json.dumps(safe_metrics, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    fi_lines = ''
    for entry in artifact['metrics'].get('feature_importances', [])[:15]:
        bar = '█' * int(entry['importance'] * 400)
        fi_lines += f"  {entry['feature'][:40]:<40} {bar} {entry['importance']:.4f}\n"

    txt_lines = [
        '═' * 55,
        ' MÉTRICAS DO MODELO — RADAR DE VIAGEM SEGURA',
        '═' * 55,
        f"  Modelo       : HistGradientBoostingClassifier",
        f"  Modo split   : {artifact['metrics']['mode']}",
        f"  Amostras     : {artifact['metrics']['samples']:,}",
        f"  Taxa positiva: {artifact['metrics']['positive_rate']:.3f}",
        '',
        '── Métricas no conjunto de teste ──────────────',
        f"  Accuracy : {artifact['metrics']['accuracy']:.4f}",
        f"  Precision: {artifact['metrics']['precision']:.4f}",
        f"  Recall   : {artifact['metrics']['recall']:.4f}",
        f"  F1-score : {artifact['metrics']['f1_score']:.4f}",
        f"  ROC-AUC  : {artifact['metrics']['roc_auc']:.4f}",
        f"  Threshold: {artifact['metrics']['threshold']:.3f}",
        '',
        '── Cross-validation (5-fold, treino) ──────────',
        f"  ROC-AUC  : {artifact['metrics']['cv_roc_auc_mean']:.4f} ± {artifact['metrics']['cv_roc_auc_std']:.4f}",
        f"  F1-score : {artifact['metrics']['cv_f1_mean']:.4f} ± {artifact['metrics']['cv_f1_std']:.4f}",
        '',
        '── Matriz de confusão ─────────────────────────',
        f"  {artifact['metrics']['confusion_matrix']}",
        '',
        '── Classification report ──────────────────────',
        artifact['metrics']['classification_report'],
        '',
        '── Feature importances (top 15) ───────────────',
        fi_lines,
    ]
    (OUTPUTS_DIR / 'relatorio_metricas.txt').write_text('\n'.join(txt_lines), encoding='utf-8')


# ── Read saved metrics ──────────────────────────────────────────
def read_metrics() -> dict | None:
    path = OUTPUTS_DIR / 'model_metrics.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))

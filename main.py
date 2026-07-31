from __future__ import annotations

from src.data_loader import load_accident_data, validate_loaded_data
from src.risk_engine import add_risk_features
from src.modeling import train_and_evaluate, save_artifacts


def main() -> None:
    df = load_accident_data()
    validation = validate_loaded_data(df)
    if not validation['ok']:
        raise RuntimeError(f"Falha de validação das bases: {validation}")
    print(f"Bases carregadas: {validation['source_counts']} | UFs na base nacional: {validation['national_uf_count']}")
    df = add_risk_features(df)
    artifact = train_and_evaluate(df)
    save_artifacts(artifact)
    print('Treino concluído. Métricas salvas em outputs/relatorio_metricas.txt')


if __name__ == '__main__':
    main()

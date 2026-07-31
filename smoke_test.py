from __future__ import annotations

from src.data_loader import load_accident_data, validate_loaded_data
from src.risk_engine import (
    add_risk_features,
    available_ufs,
    best_and_worst_hours,
    explain_trip,
    filter_dataset,
    hourly_reference,
    resolve_destination,
)


def main() -> None:
    df = load_accident_data()
    validation = validate_loaded_data(df)
    if not validation['ok']:
        raise RuntimeError(f'Falha de validação das bases: {validation}')

    df = add_risk_features(df)
    for source in ['Todas', 'Base nacional', 'Base SP']:
        filtered = filter_dataset(df, source, [], [])
        if filtered.empty:
            raise RuntimeError(f'Recorte vazio para a fonte {source}')
        print(f'[OK] Fonte {source}: {len(filtered)} registros | UFs: {len(available_ufs(df, source))}')

    sample_city = str(df['municipio'].dropna().astype(str).iloc[0])
    label, dest_df = resolve_destination(df, sample_city)
    if not label or dest_df.empty:
        raise RuntimeError('Falha ao resolver um destino conhecido da base.')

    ref = hourly_reference(dest_df, 'sexta-feira')
    picks = best_and_worst_hours(dest_df, 'sexta-feira')
    explanation = explain_trip(dest_df, 'sexta-feira', label)
    print(f'[OK] Destino: {label} | horas observadas: {int(ref["observed"].sum())}')
    print(f'[OK] Melhor janela: {picks["best_window"]} | Piores horas: {picks["worst_hours"]}')
    print(f'[OK] Explicação gerada: score {explanation["score_100"]}/100 | confiança {explanation["confidence"]}')


if __name__ == '__main__':
    main()

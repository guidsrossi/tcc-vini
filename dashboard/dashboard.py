from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from src.config import OUTPUTS_DIR, SOURCE_OPTIONS
from src.data_loader import load_accident_data, validate_loaded_data
from src.modeling import read_metrics
from src.risk_engine import (
    add_risk_features,
    available_ufs,
    best_and_worst_hours,
    build_maps_link,
    explain_trip,
    filter_dataset,
    hourly_day_matrix,
    hourly_reference,
    location_reference,
    overview_metrics,
    resolve_destination,
    road_reference,
    suggest_destinations,
    yearly_trend,
)
from src.ui import (
    apply_level_style,
    heatmap_chart,
    human_hours,
    inject_css,
    metric_card,
    offwhite_bar_chart,
    render_filters,
    render_gauge,
    render_header,
    render_journey,
    render_mini_callout,
    render_result_panel,
    render_section_heading,
    render_status_bar,
    render_top_nav,
    trend_line_chart,
)
from src.utils import human_int, pct

_PARQUET_CACHE = OUTPUTS_DIR / 'base_cache.parquet'


@st.cache_data(show_spinner=False)
def get_base() -> pd.DataFrame:
    if _PARQUET_CACHE.exists():
        try:
            return pd.read_parquet(_PARQUET_CACHE)
        except Exception:
            pass
    df = add_risk_features(load_accident_data())
    try:
        _PARQUET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_PARQUET_CACHE, index=False)
    except Exception:
        pass
    return df


@st.cache_data(show_spinner=False)
def get_validation() -> dict:
    return validate_loaded_data(load_accident_data())


@st.cache_data(show_spinner=False)
def get_filtered(source: str, years_tuple: tuple, ufs_tuple: tuple) -> pd.DataFrame:
    return filter_dataset(get_base(), source, list(years_tuple), list(ufs_tuple))


DAY_OPTIONS = [
    'segunda-feira', 'terça-feira', 'quarta-feira',
    'quinta-feira',  'sexta-feira', 'sábado', 'domingo',
]


def source_context_message(source: str) -> tuple[str, str]:
    if source == 'Todas':
        return ('Base combinada', 'Junta a base nacional com a base de São Paulo.')
    if source == 'Base nacional':
        return ('Base nacional', 'Usa apenas a base nacional — ideal para comparar estados.')
    return ('Base de São Paulo', 'Usa apenas os dados de São Paulo.')


def _add_to_history(dest: str, score: int, band: str) -> None:
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    st.session_state.search_history = [
        h for h in st.session_state.search_history if h['dest'] != dest
    ]
    st.session_state.search_history.insert(0, {'dest': dest, 'score': score, 'band': band})
    st.session_state.search_history = st.session_state.search_history[:5]


def _render_history_buttons() -> str | None:
    history = st.session_state.get('search_history', [])
    if not history:
        return None
    band_colors = {'Baixa': '#10B981', 'Moderada': '#F59E0B', 'Elevada': '#EF4444', 'Alta': '#DC2626'}
    st.markdown("<div class='hist-wrap'><span class='hist-label'>Recentes</span>", unsafe_allow_html=True)
    cols = st.columns(len(history))
    clicked = None
    for i, h in enumerate(history):
        color = band_colors.get(str(h['band']), '#5C6A80')
        with cols[i]:
            if st.button(
                f"{h['dest']} · {h['score']}/100",
                key=f"hist_{i}_{h['dest']}",
                use_container_width=True,
                type='secondary',
            ):
                clicked = h['dest']
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def _render_trip_result(df: pd.DataFrame, destination: str, day: str, origin: str = '', side: str = '') -> None:
    label, dest_df = resolve_destination(df, destination)
    if dest_df.empty:
        suggestions = suggest_destinations(df, destination)
        st.warning(f'Destino "{destination}" não encontrado. Tente outro nome.')
        if suggestions:
            render_mini_callout('?', 'Sugestões', ', '.join(suggestions))
        return

    explanation = explain_trip(dest_df, day, label)
    score = explanation['score_100']
    band  = str(explanation['band'])

    _add_to_history(str(label), score, band)

    has_comparison = bool(explanation['best_hours'])
    best_time = human_hours(explanation['best_hours'][:2]) if has_comparison else 'dados insuficientes'
    bullets = [f"Destino: {label} · {day}"]
    if explanation['worst_hours']:
        bullets.append(f"Horários para evitar: {human_hours(explanation['worst_hours'])}")
    observed_hours = int(explanation['observed_hours'])
    hour_label = 'horário' if observed_hours == 1 else 'horários'
    bullets.append(
        f"Base analisada: {human_int(explanation['records'])} registros em "
        f"{human_int(observed_hours)} {hour_label}"
    )
    if explanation['confidence'] == 'insuficiente':
        recommendation = (
            'Há poucos horários registrados para fazer uma comparação confiável. '
            'Use este resultado apenas como referência e confirme as condições atuais no Maps.'
        )
    else:
        recommendation = explanation['recommendation'].split('. ', 1)[-1]
    render_result_panel(
        f"Melhor horário: {best_time}" if has_comparison else 'Não há horários suficientes para comparar',
        f"Atenção {band.lower()} · Nota {score}/100.",
        recommendation,
        bullets,
    )

    with st.expander('Entenda o resultado e compare os horários'):
        col_gauge, col_tables = st.columns([0.9, 1.1])
        with col_gauge:
            render_section_heading('◎', 'Nota visual', 'Quanto maior, mais atenção o histórico sugere.')
            render_gauge(score, band)
            render_section_heading('i', 'Por que esse resultado')
            for reason in explanation['reasons']:
                render_mini_callout('→', 'Fator observado', reason)
        with col_tables:
            render_section_heading('◷', 'Horários do dia', 'Compare os períodos com registros disponíveis.')
            best_tbl  = explanation['best_table'].rename(columns={'hora': 'Hora', 'acidentes': 'Casos', 'support_level': 'Base', 'score_100': 'Nota', 'faixa': 'Nível', 'driver': 'Motivo'})
            worst_tbl = explanation['worst_table'].rename(columns={'hora': 'Hora', 'acidentes': 'Casos', 'support_level': 'Base', 'score_100': 'Nota', 'faixa': 'Nível', 'driver': 'Motivo'})
            tab_best, tab_worst = st.tabs(['Melhores horários', 'Horários para evitar'])
            with tab_best:
                st.dataframe(apply_level_style(best_tbl), use_container_width=True, hide_index=True)
            with tab_worst:
                st.dataframe(apply_level_style(worst_tbl), use_container_width=True, hide_index=True)

    if origin and label:
        render_route_map(origin, str(label))
    elif label:
        render_mini_callout('→', 'Rota no Maps', 'Digite a origem para abrir a rota no Google Maps.')


def render_route_map(origin: str, destination: str) -> None:
    """Embed a directions map while preserving a link to the full Maps experience."""
    origin_query = quote_plus(origin.strip())
    destination_query = quote_plus(f'{destination}, Brasil')
    embed_url = (
        'https://maps.google.com/maps?'
        f'saddr={origin_query}&daddr={destination_query}&output=embed'
    )
    with st.container(border=True):
        render_section_heading(
            '⌖',
            'Mapa da rota',
            f'Prévia de {origin.strip()} até {destination}. Confira trânsito e bloqueios antes de sair.',
        )
        st.iframe(embed_url, height=460)
        st.link_button(
            'Abrir rota completa no Google Maps',
            build_maps_link(origin, destination),
            width='stretch',
        )


def page_plan_trip(df: pd.DataFrame, source: str) -> None:
    st.markdown("<h3>Planejar viagem</h3>", unsafe_allow_html=True)

    label, copy = source_context_message(source)
    render_mini_callout('◉', label, copy)

    hist_click = _render_history_buttons()

    compare = st.toggle('Comparar dois destinos lado a lado', value=False, key='compare_toggle')

    if compare:
        st.markdown(
            "<div class='compare-header'>⇄ Modo comparação — preencha origem, destino e dia para cada rota</div>",
            unsafe_allow_html=True,
        )

        inp_a, inp_vs, inp_b = st.columns([1, 0.08, 1])

        with inp_a:
            st.markdown("<div class='compare-result-wrap compare-result-a'>", unsafe_allow_html=True)
            st.markdown("<span class='compare-side-label compare-side-a'>⬤ Rota A</span>", unsafe_allow_html=True)
            origin_a = st.text_input('Origem A', placeholder='Ex.: São Paulo, SP', key='origin_a')
            dest_a   = st.text_input('Destino A', placeholder='Ex.: Santos, SP',   key='dest_a',
                                     value=hist_click or st.session_state.get('dest_a', ''))
            day_a    = st.selectbox('Dia A', DAY_OPTIONS, index=4, key='day_a')
            st.markdown("</div>", unsafe_allow_html=True)

        with inp_vs:
            st.markdown(
                "<div style='display:flex;flex-direction:column;align-items:center;height:100%;padding-top:2.2rem;gap:.4rem'>"
                "<div style='width:1px;flex:1;background:linear-gradient(to bottom,transparent,rgba(59,130,246,.18),transparent)'></div>"
                "<div class='compare-vs-badge'>VS</div>"
                "<div style='width:1px;flex:1;background:linear-gradient(to bottom,transparent,rgba(59,130,246,.18),transparent)'></div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with inp_b:
            st.markdown("<div class='compare-result-wrap compare-result-b'>", unsafe_allow_html=True)
            st.markdown("<span class='compare-side-label compare-side-b'>⬤ Rota B</span>", unsafe_allow_html=True)
            origin_b = st.text_input('Origem B', placeholder='Ex.: São Paulo, SP', key='origin_b')
            dest_b   = st.text_input('Destino B', placeholder='Ex.: Campinas, SP', key='dest_b')
            day_b    = st.selectbox('Dia B', DAY_OPTIONS, index=4, key='day_b')
            st.markdown("</div>", unsafe_allow_html=True)

        if dest_a and dest_b:
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            res_a, res_b = st.columns(2)
            with res_a:
                st.markdown("<span class='compare-side-label compare-side-a'>⬤ Resultado — Rota A</span>", unsafe_allow_html=True)
                _render_trip_result(df, dest_a, day_a, origin=origin_a, side='a')
            with res_b:
                st.markdown("<span class='compare-side-label compare-side-b'>⬤ Resultado — Rota B</span>", unsafe_allow_html=True)
                _render_trip_result(df, dest_b, day_b, origin=origin_b, side='b')
        elif dest_a or dest_b:
            render_mini_callout('i', 'Quase lá', 'Preencha os dois destinos para ver a comparação.')
        return

    with st.container(border=True):
        render_section_heading('⌖', 'Dados da viagem', 'Informe o destino e escolha o dia para consultar o histórico.')
        c1, c2, c3 = st.columns([1.15, 1.15, 0.7])
        origin      = c1.text_input('Origem (opcional)', placeholder='Ex.: São Paulo, SP')
        default_dest = hist_click or st.session_state.get('_trip_destination', '')
        destination = c2.text_input('Destino', placeholder='Ex.: Santos, SP', value=default_dest)
        day         = c3.selectbox('Dia da semana', DAY_OPTIONS, index=4)
        calculate = st.button('Calcular melhor horário', type='primary', use_container_width=True)

    if calculate:
        if destination.strip():
            st.session_state['_trip_query'] = {
                'origin': origin.strip(), 'destination': destination.strip(), 'day': day,
            }
            st.session_state['_trip_destination'] = destination.strip()
        else:
            st.warning('Digite um destino para calcular o melhor horário.')

    if hist_click:
        st.session_state['_trip_query'] = {
            'origin': origin.strip(), 'destination': hist_click, 'day': day,
        }

    query = st.session_state.get('_trip_query')
    if not query:
        render_journey()
        return

    _render_trip_result(
        df, query['destination'], query['day'], origin=query.get('origin', '')
    )

    if not query.get('origin'):
        render_mini_callout('→', 'Próximo passo', 'Digite a origem para abrir a rota no Google Maps.')


def page_overview(df: pd.DataFrame) -> None:
    st.markdown("<h3>Resumo rápido</h3>", unsafe_allow_html=True)
    metrics = overview_metrics(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card('Casos no filtro',  human_int(metrics['accidents']),
                    tooltip='Total de registros de acidentes no conjunto filtrado atualmente.')
    with c2:
        metric_card('Casos graves',     pct(metrics['serious_rate']),
                    tooltip='Proporção de acidentes classificados como graves no filtro atual.')
    with c3:
        metric_card('Mortes',           human_int(metrics['deaths']),
                    tooltip='Total de óbitos registrados nos dados filtrados.')
    with c4:
        metric_card('Nota média',       f"{int(round(metrics['score']))}/100",
                    tooltip='Nota média de atenção calculada sobre o conjunto filtrado (0 = baixo risco, 100 = alto).')

    picks = best_and_worst_hours(df)
    render_result_panel(
        f"Melhor horário geral: {human_hours(picks['best_window'][:2])}",
        'Este resumo mostra, de forma simples, quais horários parecem melhores ou piores.',
        'Para uma resposta mais útil, vá em Planejar viagem e digite o destino.',
        [
            f"Horários que pedem mais cuidado: {human_hours(picks['worst_hours'])}",
            'A nota leva em conta quantidade de casos e gravidade.',
        ],
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown(
            "<div class='card'><div class='card-header'><div class='card-icon'>!</div>"
            "<div class='section-title'>Cidades que pedem mais atenção</div></div>",
            unsafe_allow_html=True,
        )
        loc = location_reference(df).head(10)
        display = loc[['municipio', 'uf', 'acidentes', 'support_level', 'faixa', 'score_100', 'principal_driver']].rename(
            columns={'municipio': 'Município', 'uf': 'UF', 'acidentes': 'Casos',
                     'support_level': 'Base', 'faixa': 'Nível', 'score_100': 'Nota', 'principal_driver': 'Motivo principal'}
        )
        st.dataframe(apply_level_style(display), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            "<div class='card'><div class='card-header'><div class='card-icon'>◷</div>"
            "<div class='section-title'>Notas por hora</div></div>",
            unsafe_allow_html=True,
        )
        hours = hourly_reference(df)
        st.plotly_chart(offwhite_bar_chart(hours, 'hora', 'score_100', title='Nota por hora do dia'), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    trend = yearly_trend(df)
    if not trend.empty and len(trend) > 1:
        st.markdown(
            "<div class='card'><div class='card-header'><div class='card-icon'>↗</div>"
            "<div class='section-title'>Tendência por ano</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(trend_line_chart(trend), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)


def page_dangerous_areas(df: pd.DataFrame) -> None:
    st.markdown("<h3>Lugares que pedem mais atenção</h3>", unsafe_allow_html=True)
    st.caption('Cidades e rodovias com maior nota de atenção nos dados históricos.')
    loc   = location_reference(df)
    roads = road_reference(df)

    a, b = st.columns([1.15, 0.85])
    with a:
        st.markdown(
            "<div class='card'><div class='card-header'><div class='card-icon'>◎</div>"
            "<div class='section-title'>Cidades no topo da atenção</div></div>",
            unsafe_allow_html=True,
        )
        display = loc[['municipio', 'uf', 'acidentes', 'support_level', 'score_100', 'faixa', 'principal_driver']].head(25).rename(
            columns={'municipio': 'Município', 'uf': 'UF', 'acidentes': 'Casos',
                     'support_level': 'Base', 'score_100': 'Nota', 'faixa': 'Nível', 'principal_driver': 'Motivo principal'}
        )
        st.dataframe(apply_level_style(display), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        render_result_panel(
            'Como entender a lista',
            'A nota não prevê acidente. Ela mostra onde os dados históricos pedem mais cuidado.',
            'Se seu destino aparecer no topo, use Planejar viagem para achar um horário melhor.',
            [
                'Quanto maior a nota, mais atenção vale ter.',
                'A coluna Base mostra a quantidade de dados disponíveis.',
            ],
        )

    st.markdown(
        "<div class='card'><div class='card-header'><div class='card-icon'>≡</div>"
        "<div class='section-title'>Rodovias que pedem mais atenção</div></div>",
        unsafe_allow_html=True,
    )
    roads_display = roads[['rodovia', 'acidentes', 'support_level', 'score_100', 'faixa', 'principal_driver']].head(20).rename(
        columns={'rodovia': 'Rodovia', 'acidentes': 'Casos', 'support_level': 'Base',
                 'score_100': 'Nota', 'faixa': 'Nível', 'principal_driver': 'Motivo principal'}
    )
    st.dataframe(apply_level_style(roads_display), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def page_best_hours(df: pd.DataFrame) -> None:
    st.markdown("<h3>Melhores horários para sair</h3>", unsafe_allow_html=True)
    day   = st.selectbox('Escolha o dia', DAY_OPTIONS, index=4, key='best_hours_day')
    ref   = hourly_reference(df, day)
    picks = best_and_worst_hours(df, day)

    render_result_panel(
        f"Melhor horário em {day}: {human_hours(picks['best_window'])}",
        'Esta tela compara as 24 horas do dia e só mostra resultado quando há dados.',
        'Use a lista abaixo para escolher a hora de sair.',
        [f"Horários para evitar: {human_hours(picks['worst_hours'])}"],
    )

    a, b = st.columns([1.05, 0.95])
    with a:
        st.markdown(
            "<div class='card'><div class='card-header'><div class='card-icon'>◷</div>"
            "<div class='section-title'>Nota por hora</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            offwhite_bar_chart(ref, 'hora', 'score_100', title=f'Notas — {day}'),
            use_container_width=True, config={'displayModeBar': False},
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown(
            "<div class='card'><div class='card-header'><div class='card-icon'>↕</div>"
            "<div class='section-title'>Ranking completo</div></div>",
            unsafe_allow_html=True,
        )
        rank = ref[['hora', 'observed', 'acidentes', 'support_level', 'score_100', 'faixa', 'driver']].copy()
        rank['_ord'] = rank['observed'].astype(int)
        rank = rank.sort_values(['_ord', 'score_100', 'acidentes', 'hora'], ascending=[False, True, False, True])
        rank = rank.drop(columns=['_ord', 'observed']).rename(
            columns={'hora': 'Hora', 'acidentes': 'Casos', 'support_level': 'Base',
                     'score_100': 'Nota', 'faixa': 'Nível', 'driver': 'Motivo principal'}
        )
        st.dataframe(apply_level_style(rank), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    matrix = hourly_day_matrix(df)
    if not matrix.empty:
        st.markdown(
            "<div class='card'><div class='card-header'><div class='card-icon'>⊞</div>"
            "<div class='section-title'>Mapa de calor — dia × hora</div>"
            "<div class='section-copy' style='margin:0'>Verde = menor atenção · Vermelho = maior atenção histórica</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(heatmap_chart(matrix), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)


def page_map(df: pd.DataFrame) -> None:
    st.markdown("<h3>Mapa dos registros</h3>", unsafe_allow_html=True)
    geo = df.dropna(subset=['latitude', 'longitude'])[
        ['latitude', 'longitude', 'municipio', 'uf', 'acidente_grave']
    ].copy()
    # Remove coordenadas incompatíveis com a malha rodoviária brasileira.
    # Esses registros continuam na base e nas métricas, mas não são desenhados
    # no mapa para evitar pontos espúrios no oceano ou em outros continentes.
    geo = geo[
        geo['latitude'].between(-34.0, 6.0)
        & geo['longitude'].between(-74.0, -34.0)
    ].copy()
    if geo.empty:
        st.warning('Não há pontos no mapa para esse filtro.')
        return
    available_points = len(geo)
    if available_points > 12000:
        geo = geo.sample(12000, random_state=42)
    displayed_points = len(geo)

    st.markdown(
        f"""
        <div class='map-legend'>
            <strong>{displayed_points:,} de {available_points:,} registros exibidos</strong>
            <span class='legend-item'><span class='legend-dot legend-red'></span>Ponto de acidente</span>
            <span>Aproxime o mapa para separar os pontos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        import pydeck as pdk
        severity = pd.to_numeric(geo['acidente_grave'], errors='coerce')
        geo['categoria'] = severity.map(
            {1: 'Acidente grave', 0: 'Acidente sem vítima grave'}
        ).fillna('Gravidade não informada')
        geo['cor'] = [[239, 68, 68, 255] for _ in range(len(geo))]
        geo['titulo'] = geo['municipio'].fillna('Local não informado') + ', ' + geo['uf'].fillna('')
        lat_span = float(geo['latitude'].max() - geo['latitude'].min())
        lon_span = float(geo['longitude'].max() - geo['longitude'].min())
        span = max(lat_span, lon_span)
        zoom = 4.1 if span > 25 else 4.8 if span > 10 else 5.8 if span > 4 else 7.2 if span > 1.5 else 9
        is_national_view = span > 25
        view = pdk.ViewState(
            latitude=-14.2 if is_national_view else float(geo['latitude'].median()),
            longitude=-51.9 if is_national_view else float(geo['longitude'].median()),
            zoom=zoom,
            pitch=0,
        )
        layers = [pdk.Layer(
            'ScatterplotLayer', data=geo,
            get_position='[longitude, latitude]',
            get_fill_color='cor',
            get_line_color='[255, 255, 255, 210]',
            get_radius=80,
            radius_units='meters',
            radius_min_pixels=.45,
            radius_max_pixels=4,
            stroked=False,
            pickable=True,
            auto_highlight=True,
            opacity=1,
        )]
        deck  = pdk.Deck(
            layers=layers, initial_view_state=view,
            map_style='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
            tooltip={
                'html': (
                    '<div><div style="color:#94A3B8;font-size:10px;font-weight:800;'
                    'letter-spacing:.08em;margin-bottom:5px">LOCAL DO REGISTRO</div>'
                    '<div style="color:#F8FAFC;font-size:14px;font-weight:800">'
                    '{titulo}</div>'
                    '<div style="color:#CBD5E1;font-size:12px;margin-top:5px">'
                    '{categoria}</div></div>'
                ),
                'style': {
                    'backgroundColor': '#111827',
                    'color': '#E5E7EB',
                    'border': '1px solid rgba(148, 163, 184, .3)',
                    'borderRadius': '10px',
                    'padding': '12px 14px',
                    'fontFamily': 'Inter, sans-serif',
                    'fontSize': '13px',
                    'boxShadow': '0 10px 30px rgba(0, 0, 0, .45)',
                },
            },
        )
        st.pydeck_chart(deck, width='stretch', height=520)
    except Exception:
        st.map(geo[['latitude', 'longitude']], use_container_width=True)


def page_table(df: pd.DataFrame) -> None:
    st.markdown("<h3>Tabela completa</h3>", unsafe_allow_html=True)
    cols      = ['data_inversa', 'uf', 'municipio', 'rodovia', 'tipo_acidente', 'causa_acidente', 'condicao_metereologica', 'mortos', 'feridos_graves', 'fonte_base']
    available = [c for c in cols if c in df.columns]
    view      = df[available].sort_values('data_inversa', ascending=False)

    st.markdown(
        "<div class='card'><div class='card-header'><div class='card-icon'>≡</div>"
        f"<div><div class='section-title'>Registros filtrados</div>"
        f"<div class='section-copy' style='margin:0'>Mostrando {min(1000, len(view)):,} dos {len(view):,} registros no filtro.</div></div></div>",
        unsafe_allow_html=True,
    )
    st.dataframe(view.head(1000), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.download_button(
        'Baixar dados filtrados em CSV',
        view.to_csv(index=False).encode('utf-8'),
        file_name='recorte_radar_viagem_segura.csv',
        mime='text/csv',
        use_container_width=True,
    )


def page_about(df: pd.DataFrame) -> None:
    st.markdown("<h3>Sobre o sistema</h3>", unsafe_allow_html=True)

    validation = get_validation()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card('Total de registros',      human_int(validation['rows']),
                    tooltip='Soma de todos os registros carregados de todas as bases.')
    with c2:
        metric_card('Estados (base nacional)', str(validation['national_uf_count']),
                    tooltip='Número de UFs distintas na base nacional de acidentes.')
    with c3:
        metric_card('Estrutura dos dados',     'Ok' if validation['ok'] else 'Problema',
                    tooltip='Indica se todas as colunas esperadas estão presentes nos dados.')

    if validation['missing_columns']:
        st.error(f"Faltam estas colunas: {', '.join(validation['missing_columns'])}")

    st.markdown(
        "<div class='card'><div class='card-header'><div class='card-icon'>?</div>"
        "<div class='section-title'>Como o sistema funciona</div></div>",
        unsafe_allow_html=True,
    )
    render_result_panel(
        'Como a nota de atenção é calculada',
        'A nota junta três coisas: quantidade de casos, gravidade e horário. Depois o sistema organiza horários, cidades e rodovias do melhor para o pior.',
        'O sistema só recomenda horários com dados e avisa quando há poucos disponíveis.',
        [
            'Nota de 0 a 100: quanto maior, mais atenção vale ter.',
            'Níveis: Baixa, Moderada, Elevada e Alta.',
            'Confiança: depende da quantidade de dados usada.',
        ],
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "<div class='card'><div class='card-header'><div class='card-icon'>▣</div>"
        "<div class='section-title'>Composição dos dados</div></div>",
        unsafe_allow_html=True,
    )
    counts = df['fonte_base'].value_counts().rename_axis('Base').reset_index(name='Acidentes')
    st.dataframe(counts, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    metrics = read_metrics()
    if metrics:
        st.markdown(
            "<div class='card'><div class='card-header'><div class='card-icon'>◈</div>"
            "<div class='section-title'>Métricas do modelo (parte técnica)</div></div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            metric_card('Accuracy',  f"{metrics['accuracy']:.3f}",
                        tooltip='Proporção de previsões corretas do modelo de risco.')
        with m2:
            metric_card('Precision', f"{metrics['precision']:.3f}",
                        tooltip='De todos os casos marcados como graves, quantos realmente eram.')
        with m3:
            metric_card('Recall',    f"{metrics['recall']:.3f}",
                        tooltip='De todos os casos graves reais, quantos o modelo capturou.')
        with m4:
            metric_card('F1-score',  f"{metrics['f1_score']:.3f}",
                        tooltip='Média harmônica entre Precision e Recall.')
        with m5:
            metric_card('ROC-AUC',   f"{metrics.get('roc_auc', 0):.3f}",
                        tooltip='Área sob a curva ROC. Quanto mais próximo de 1.0, melhor a separação das classes.')

        if metrics.get('cv_roc_auc_mean'):
            st.markdown(
                "<div class='card'><div class='card-header'><div class='card-icon'>↻</div>"
                "<div class='section-title'>Cross-validation (5-fold)</div></div>",
                unsafe_allow_html=True,
            )
            cv1, cv2 = st.columns(2)
            with cv1:
                metric_card(
                    'ROC-AUC médio (CV)',
                    f"{metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}",
                    tooltip='Média e desvio padrão do ROC-AUC nos 5 folds de validação cruzada no treino.',
                )
            with cv2:
                metric_card(
                    'F1 médio (CV)',
                    f"{metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}",
                    tooltip='Média e desvio padrão do F1-score nos 5 folds de validação cruzada no treino.',
                )
            st.markdown('</div>', unsafe_allow_html=True)

        fi = metrics.get('feature_importances', [])
        if fi:
            import plotly.graph_objects as go
            st.markdown(
                "<div class='card'><div class='card-header'><div class='card-icon'>◇</div>"
                "<div class='section-title'>Importância das features (top 15)</div></div>",
                unsafe_allow_html=True,
            )
            fi_top = fi[:15]
            names  = [e['feature'].replace('cat__', '').replace('cyclic__', '⟳ ').replace('num__', '') for e in fi_top]
            values = [e['importance'] for e in fi_top]
            fig = go.Figure(go.Bar(
                x=values[::-1], y=names[::-1],
                orientation='h',
                marker=dict(
                    color=[f'rgba(59,130,246,{0.45 + 0.55 * v / max(values)})' for v in values[::-1]],
                    line=dict(color='rgba(96,165,250,0.40)', width=1),
                    cornerradius=4,
                ),
                hovertemplate='<b>%{y}</b><br>Importância: %{x:.4f}<extra></extra>',
                text=[f'{v:.3f}' for v in values[::-1]],
                textposition='outside',
                textfont=dict(size=9, color='#5C6A80'),
            ))
            fig.update_layout(
                height=420, margin=dict(l=8, r=40, t=8, b=8),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#5C6A80', size=9), zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(color='#E2E8F4', size=10)),
                font=dict(color='#E2E8F4', family='Inter, sans-serif'),
                hoverlabel=dict(bgcolor='rgba(11,15,24,0.97)', bordercolor='rgba(59,130,246,0.30)', font=dict(color='#E2E8F4')),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        render_mini_callout('i', 'Importante', 'Essas métricas são técnicas. O que mais importa para uso é a nota de atenção e a lista de horários.')
        st.markdown('</div>', unsafe_allow_html=True)


def page_about_accessible(df: pd.DataFrame) -> None:
    """Explain the project in plain language, with technical details on demand."""
    st.markdown("<h3>Entenda o Radar</h3>", unsafe_allow_html=True)
    st.caption('Uma explicação simples sobre o que o sistema mostra — e o que ele não consegue prever.')

    with st.container(border=True):
        render_section_heading('◎', 'O que este sistema faz')
        st.markdown(
            """
            O Radar analisa **acidentes que já aconteceram** e procura padrões de horário,
            local e gravidade. Quando você informa um destino, ele compara esses padrões e
            sugere períodos que historicamente tiveram resultados melhores ou piores.
            """
        )
        st.info(
            'A nota é uma referência histórica, não uma previsão. Use também um aplicativo '
            'de trânsito atualizado, confira o clima e respeite a sinalização da via.',
            icon='ℹ️',
        )

    st.markdown("<h3>Como ler o resultado</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            render_section_heading('1', 'Nota de atenção')
            st.markdown('Vai de **0 a 100**. Quanto maior, mais sinais históricos pedem cuidado.')
    with c2:
        with st.container(border=True):
            render_section_heading('2', 'Nível')
            st.markdown('Traduz a nota em **Baixa, Moderada, Elevada ou Alta** para facilitar a leitura.')
    with c3:
        with st.container(border=True):
            render_section_heading('3', 'Confiança')
            st.markdown('Indica a quantidade de dados por trás da resposta. Mais registros dão uma referência mais estável.')

    with st.container(border=True):
        render_section_heading('✓', 'O que entra no cálculo')
        a, b = st.columns(2)
        with a:
            st.markdown(
                """
                - quantidade de acidentes registrados;
                - mortos e feridos graves;
                - horário e dia da semana;
                - clima informado no registro.
                """
            )
        with b:
            st.markdown(
                """
                - município e rodovia;
                - causa e tipo do acidente;
                - tipo e traçado da pista;
                - quantidade de dados disponíveis.
                """
            )

    validation = get_validation()
    with st.container(border=True):
        render_section_heading('▣', 'Dados disponíveis', 'Resumo das informações carregadas nesta consulta.')
        m1, m2, m3 = st.columns(3)
        m1.metric('Acidentes analisados', human_int(validation['rows']))
        m2.metric('Estados na base nacional', str(validation['national_uf_count']))
        m3.metric('Verificação dos dados', 'Tudo certo' if validation['ok'] else 'Revisão necessária')
        counts = df['fonte_base'].value_counts().rename_axis('Fonte').reset_index(name='Registros')
        st.dataframe(counts, use_container_width=True, hide_index=True)

    with st.expander('Ver detalhes técnicos do modelo'):
        metrics = read_metrics()
        if not metrics:
            st.write('As métricas técnicas ainda não estão disponíveis.')
        else:
            st.markdown(
                'Esta parte é voltada a estudantes e avaliadores. Ela mostra como o modelo '
                'se comportou em dados separados para teste.'
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Acertos gerais', f"{metrics['accuracy']:.1%}")
            m2.metric('Casos graves encontrados', f"{metrics['recall']:.1%}")
            m3.metric('Precisão dos alertas', f"{metrics['precision']:.1%}")
            m4.metric('Equilíbrio F1', f"{metrics['f1_score']:.1%}")
            st.caption(
                'O modelo prioriza encontrar mais casos graves. Por isso, ele pode emitir '
                'mais alertas que depois não se confirmam como graves.'
            )

    st.warning(
        'Não use o Radar como única fonte para decidir uma viagem. Ele não recebe trânsito, '
        'obras, bloqueios ou clima em tempo real.',
        icon='⚠️',
    )


def main() -> None:
    st.set_page_config(page_title='Radar de Viagem Segura', page_icon='🧭', layout='wide')
    inject_css()
    render_header()
    page = render_top_nav()

    with st.spinner('Carregando dados...'):
        base = get_base()

    years = sorted([int(y) for y in base['ano_referencia'].dropna().unique().tolist()])
    ufs_by_source = {option: available_ufs(base, option) for option in SOURCE_OPTIONS}
    source, selected_years, selected_ufs, ufs_for_source = render_filters(
        SOURCE_OPTIONS, years, ufs_by_source
    )

    filtered = get_filtered(source, tuple(selected_years), tuple(selected_ufs))
    counts   = filtered['fonte_base'].value_counts().to_dict() if 'fonte_base' in filtered.columns else {}
    render_status_bar(len(filtered), source, counts, len(ufs_for_source))

    if filtered.empty:
        st.warning('Não há dados com esse filtro. Tente mudar a base, os anos ou os estados.')
        return

    page_map_fn = {
        'Planejar viagem':          lambda frame: page_plan_trip(frame, source),
        'Panorama rápido':          page_overview,
        'Lugares com mais atenção': page_dangerous_areas,
        'Melhores horários':        page_best_hours,
        'Mapa histórico':           page_map,
        'Tabela':                   page_table,
        'Sobre':                    page_about_accessible,
    }
    page_map_fn[page](filtered)


if __name__ == '__main__':
    main()

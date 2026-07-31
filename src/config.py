from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'dados'
OUTPUTS_DIR = ROOT_DIR / 'outputs'
MODELS_DIR = ROOT_DIR / 'modelos'

APP_TITLE = 'Radar de Viagem Segura'
APP_SUBTITLE = 'Veja os melhores horários para viajar com mais tranquilidade.'
APP_DESCRIPTION = (
    'O sistema usa dados do passado para mostrar horários que parecem melhores ou piores para viajar. '
    'Ele ajuda na escolha, mas não adivinha o futuro e não substitui aplicativos de navegação.'
)

PAGES = [
    ('Planejar viagem', 'Digite origem, destino e dia. Veja o melhor horário e abra a rota no Maps.'),
    ('Panorama rápido', 'Veja um resumo simples dos dados antes de decidir.'),
    ('Lugares com mais atenção', 'Veja cidades e rodovias que pedem mais atenção.'),
    ('Melhores horários', 'Compare os horários que parecem melhores e piores para viajar.'),
    ('Mapa histórico', 'Veja no mapa onde há mais registros.'),
    ('Tabela', 'Veja a tabela completa e baixe os dados filtrados.'),
    ('Sobre', 'Entenda de forma simples como o sistema funciona.'),
]

SOURCE_OPTIONS = ['Todas', 'Base nacional', 'Base SP']

THEME = {
    'bg': '#050709',
    'bg_2': '#080C12',
    'surface': 'rgba(11,15,24,0.98)',
    'surface_strong': '#0C1018',
    'surface_soft': '#090D16',
    'text': '#E2E8F4',
    'muted': '#5C6A80',
    'border': 'rgba(255,255,255,0.07)',
    'accent': '#3B82F6',
    'accent_soft': 'rgba(59,130,246,0.12)',
    'accent_2': '#60A5FA',
    'danger': '#EF4444',
    'ok': '#10B981',
    'shadow': '0 24px 56px rgba(0,0,0,0.70)',
}

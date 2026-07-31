import types, sys
import pandas as pd

# Minimal streamlit stub so the UI module can import.
sys.modules.setdefault('streamlit', types.SimpleNamespace(markdown=lambda *a, **k: None))

from src.ui import offwhite_bar_chart

df = pd.DataFrame({
    'hora': [0, 1, 2],
    'score_100': [10.0, 20.0, 30.0],
    'observed': [True, True, False],
})

fig = offwhite_bar_chart(df, 'hora', 'score_100', title='Teste')
print(type(fig))
print('is_figure =', hasattr(fig, 'to_dict'))

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import APP_DESCRIPTION, APP_SUBTITLE, APP_TITLE, PAGES, THEME


CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&display=swap');

:root {
    --bg:               __BG__;
    --bg2:              __BG_2__;
    --surface:          __SURFACE__;
    --surface-strong:   __SURFACE_STRONG__;
    --surface-soft:     __SURFACE_SOFT__;
    --text:             __TEXT__;
    --muted:            __MUTED__;
    --border:           __BORDER__;
    --accent:           __ACCENT__;
    --accent-soft:      __ACCENT_SOFT__;
    --accent-2:         __ACCENT_2__;
    --danger:           __DANGER__;
    --ok:               __OK__;
    --shadow:           __SHADOW__;
    --shadow-card:      0 8px 32px rgba(0,0,0,0.55);
    --ring:             0 0 0 3px rgba(59,130,246,0.25);
    --r:                16px;
    --r-sm:             10px;
    --r-xs:             8px;
}

@keyframes fadeUp   { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn   { from{opacity:0} to{opacity:1} }
@keyframes blip     { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }
@keyframes scanline {
    0%   { background-position: 0 -100vh; }
    100% { background-position: 0 100vh; }
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -.01em;
}

/* ── APP BG ────────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -5%,  rgba(59,130,246,0.07), transparent),
        radial-gradient(ellipse 50% 40% at 90% 90%,  rgba(59,130,246,0.04), transparent),
        radial-gradient(rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 100%, 100%, 28px 28px;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1240px;
    padding: 1.25rem 2rem 4.5rem;
}

[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 1.15rem; }
[data-testid="stHorizontalBlock"] { gap: 1rem; }
[data-testid="column"] > [data-testid="stVerticalBlock"] { gap: .85rem; }

/* ── SCROLLBAR ─────────────────────────────────────────────── */
::-webkit-scrollbar              { width:5px; height:5px; }
::-webkit-scrollbar-track        { background: transparent; }
::-webkit-scrollbar-thumb        { background: rgba(255,255,255,0.08); border-radius:999px; }
::-webkit-scrollbar-thumb:hover  { background: rgba(59,130,246,0.40); }

/* ── HERO ──────────────────────────────────────────────────── */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.15rem 2.25rem 1.9rem;
    border-radius: 24px;
    border: 1px solid rgba(59,130,246,0.16);
    background: linear-gradient(135deg, #0B1120 0%, #090D18 100%);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.03), var(--shadow);
    margin-bottom: .35rem;
    animation: fadeUp .55s cubic-bezier(.22,.68,0,1.2);
    isolation: isolate;
}
.hero::before {
    content:'';
    position:absolute; top:-40%; right:-8%; width:500px; height:500px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(59,130,246,0.10) 0%, transparent 65%);
    pointer-events:none; z-index:-1;
}
.hero::after {
    content:'';
    position:absolute; bottom:-50%; left:-5%; width:360px; height:360px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(59,130,246,0.05) 0%, transparent 70%);
    pointer-events:none; z-index:-1;
}
.hero-topline {
    display:flex; align-items:center; justify-content:space-between;
    gap:1rem; flex-wrap:wrap; margin-bottom:1.3rem;
}
.hero-eyebrow {
    display:inline-flex; align-items:center; gap:.5rem;
    color:var(--accent-2); font-size:.73rem; font-weight:800;
    letter-spacing:.09em; text-transform:uppercase;
}
.hero-eyebrow::before { content:''; width:18px; height:2px; border-radius:99px; background:var(--accent); }
.hero-meta {
    display:flex; align-items:center; gap:.7rem; flex-wrap:wrap;
    color:var(--muted); font-size:.76rem; font-weight:650;
}
.hero-meta span { display:inline-flex; align-items:center; gap:.35rem; }
.hero-meta span::before { content:'•'; color:var(--accent); }
.pill {
    display:inline-flex; align-items:center; gap:.45rem;
    padding:.3rem .75rem; border-radius:999px;
    border:1px solid rgba(59,130,246,0.28);
    background: rgba(59,130,246,0.10);
    color: #60A5FA;
    font-size:.72rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase;
}
.pill-dot {
    width:6px; height:6px; border-radius:50%;
    background:#3B82F6;
    animation: blip 1.8s ease-in-out infinite;
}
.hero-kpis { display:flex; gap:.55rem; flex-wrap:wrap; justify-content:flex-end; }
.hero-kpi {
    padding:.6rem 1rem; border-radius:var(--r-sm);
    border:1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.03);
    min-width:130px; backdrop-filter:blur(4px);
}
.hero-kpi-label {
    display:block; font-size:.68rem; text-transform:uppercase;
    letter-spacing:.09em; color:var(--muted); margin-bottom:.22rem; font-weight:700;
}
.hero-kpi-value { font-size:.92rem; font-weight:700; color:var(--text); }

.hero h1 {
    margin:0 0 .7rem; font-size:clamp(2.25rem, 4.2vw, 3.25rem); font-weight:900;
    line-height:1.02; letter-spacing:-.055em; color:#FFFFFF; max-width:18ch;
}
.hero h1 em { font-style:normal; color:var(--accent-2); }
.hero-sub { color:#91A1B8; font-size:.96rem; line-height:1.7; max-width:760px; }

/* ── STATUS BAR ────────────────────────────────────────────── */
.status-bar {
    display:flex; align-items:center; gap:.6rem; flex-wrap:wrap;
    padding:.7rem 1rem; border-radius:12px;
    border:1px solid var(--border);
    background: rgba(255,255,255,0.025);
    margin:0 0 .4rem;
    animation: fadeIn .4s ease;
}
.status-chip {
    display:inline-flex; align-items:center; gap:.35rem;
    font-size:.76rem; font-weight:700; color:var(--muted);
}
.status-chip strong { color:var(--text); font-weight:800; }
.status-sep { color:rgba(255,255,255,0.10); font-size:.8rem; }

/* ── NAV ───────────────────────────────────────────────────── */
.nav-wrap {
    position:sticky; top:.5rem; z-index:30;
    padding:.45rem .5rem; margin:0 0 .15rem;
    border-radius:var(--r);
    border:1px solid rgba(255,255,255,0.05);
    background: rgba(5,7,9,0.88);
    backdrop-filter:blur(18px);
    box-shadow:0 4px 28px rgba(0,0,0,0.60), 0 0 0 1px rgba(255,255,255,0.04);
    animation: fadeUp .45s ease-out;
}
div[role="radiogroup"] { gap:.3rem !important; }
div[data-testid="stRadio"] > label { display:none !important; }
div[data-testid="stRadio"] {
    position:sticky; top:.65rem; z-index:30;
    padding:.42rem .48rem; border-radius:14px;
    border:1px solid rgba(148,163,184,.10);
    background:rgba(5,7,9,.91); backdrop-filter:blur(18px);
    box-shadow:0 10px 30px rgba(0,0,0,.42);
}
div[data-testid="stRadio"] div[role="radiogroup"] { gap:.35rem !important; flex-wrap:wrap; }
div[data-testid="stRadio"] div[role="radiogroup"] label {
    padding:.58rem .85rem !important; border:1px solid transparent !important;
    border-radius:10px !important; background:transparent !important;
    cursor:pointer !important; transition:background .15s,border-color .15s,color .15s !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display:none !important; }
div[data-testid="stRadio"] div[role="radiogroup"] label p {
    color:var(--muted) !important; font-size:.84rem !important; font-weight:750 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background:rgba(59,130,246,.14) !important; border-color:rgba(59,130,246,.26) !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color:var(--accent-2) !important; }
div[role="radiogroup"] label[data-baseweb="radio"] {
    border-radius:var(--r-sm) !important; padding:.46rem .82rem !important;
    border:1px solid transparent !important; background:transparent !important;
    transition:background .14s, border-color .14s, transform .14s;
}
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display:none; }
div[role="radiogroup"] label[data-baseweb="radio"] p {
    font-size:.86rem; font-weight:700; color:var(--muted);
}
div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] {
    background: rgba(59,130,246,0.14) !important;
    border-color: rgba(59,130,246,0.28) !important;
}
div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] p { color:#60A5FA !important; }
div[role="radiogroup"] label[data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    transform: translateY(-1px);
}

/* ── CARDS ─────────────────────────────────────────────────── */
.panel, .card {
    background: linear-gradient(145deg, rgba(13,18,29,.98), rgba(9,13,22,.98));
    border:1px solid rgba(148,163,184,.11);
    border-radius:18px;
    box-shadow: 0 14px 42px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.025);
    padding:1.4rem 1.45rem;
    margin-bottom:.45rem;
    animation: fadeUp .5s ease-out;
}
.card-header {
    display:flex; align-items:center; gap:.6rem;
    margin-bottom:.85rem; padding-bottom:.75rem;
    border-bottom:1px solid rgba(255,255,255,0.05);
}
.card-icon {
    width:2rem; height:2rem; border-radius:var(--r-xs);
    display:flex; align-items:center; justify-content:center;
    background: var(--accent-soft);
    border:1px solid rgba(59,130,246,0.20);
    color:var(--accent-2); font-size:.9rem; font-weight:900;
    flex-shrink:0;
}
.section-title {
    font-size:.98rem; font-weight:800;
    letter-spacing:-.02em; color:var(--text); margin-bottom:.2rem;
}
.section-copy {
    font-size:.86rem; color:var(--muted);
    line-height:1.62; margin-bottom:.85rem;
}
.muted { color:var(--muted); font-size:.86rem; }

/* ── KPI GRID ──────────────────────────────────────────────── */
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; }

.kpi {
    position:relative; overflow:hidden;
    padding:1.25rem 1.2rem 1.15rem;
    min-height:122px;
    background: var(--surface);
    border:1px solid var(--border);
    border-radius:var(--r);
    box-shadow: 0 12px 34px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.025);
    transition: transform .18s, border-color .18s, box-shadow .18s;
    animation: fadeUp .5s ease-out;
}
.kpi::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
}
.kpi::after {
    content:''; position:absolute; top:0; right:0; width:80px; height:80px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(59,130,246,0.06), transparent 70%);
    pointer-events:none;
}
.kpi:hover {
    transform: translateY(-3px);
    border-color: rgba(59,130,246,0.22);
    box-shadow: 0 18px 40px rgba(0,0,0,0.60), 0 0 0 1px rgba(59,130,246,0.10);
}
.kpi-label {
    font-size:.71rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.08em; color:var(--muted); margin-bottom:.5rem;
}
.kpi-value {
    font-size:1.85rem; font-weight:900;
    line-height:1.0; letter-spacing:-.05em; color:var(--text);
}
.kpi-help { font-size:.76rem; color:var(--muted); margin-top:.45rem; line-height:1.45; }

/* ── JOURNEY STEPS ─────────────────────────────────────────── */
.journey-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; }
.journey-step {
    position:relative; padding:1.15rem 1.1rem;
    background: var(--surface);
    border:1px solid var(--border);
    border-radius:var(--r);
    box-shadow: var(--shadow-card);
    min-height:155px;
    transition: transform .18s, border-color .18s, box-shadow .18s;
    animation: fadeUp .55s ease-out;
}
.journey-step:hover {
    transform: translateY(-3px);
    border-color: rgba(59,130,246,0.22);
    box-shadow: 0 18px 40px rgba(0,0,0,0.60);
}
.step-index {
    display:inline-flex; width:1.7rem; height:1.7rem; border-radius:var(--r-xs);
    align-items:center; justify-content:center;
    background: rgba(59,130,246,0.14);
    border:1px solid rgba(59,130,246,0.22);
    color:#60A5FA; font-size:.74rem; font-weight:900;
    margin-bottom:.8rem;
}
.journey-step strong {
    display:block; font-size:.96rem; font-weight:800;
    color:var(--text); margin-bottom:.3rem;
}

/* ── RESULT HERO ───────────────────────────────────────────── */
.result-hero {
    position:relative; overflow:hidden;
    padding:1.4rem 1.5rem;
    margin:.3rem 0 1rem;
    border-radius:var(--r);
    border:1px solid rgba(59,130,246,0.18);
    background: linear-gradient(135deg, rgba(15,22,40,0.99) 0%, rgba(10,14,28,0.98) 100%);
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.04);
    animation: fadeUp .5s ease-out;
}
.result-hero::before {
    content:''; position:absolute; top:-30%; right:-5%; width:300px; height:300px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(59,130,246,0.09), transparent 65%);
    pointer-events:none;
}
.result-tag {
    display:inline-block; padding:.28rem .62rem; border-radius:999px;
    background: rgba(59,130,246,0.14); border:1px solid rgba(59,130,246,0.26);
    color:#60A5FA; font-size:.70rem; font-weight:800;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:.65rem;
}
.result-title {
    font-size:1.52rem; font-weight:900;
    letter-spacing:-.04em; margin:0 0 .42rem; color:#FFFFFF;
}
.result-copy { font-size:.92rem; line-height:1.66; color:var(--text); }
.subtle-line  { height:1px; background:rgba(255,255,255,0.06); margin:.9rem 0; }
.result-note  { color:var(--muted); line-height:1.58; font-size:.88rem; margin-top:.7rem; }
.result-list  { margin:.85rem 0 0; padding-left:1rem; color:var(--muted); font-size:.86rem; }
.result-list li { margin-bottom:.3rem; }

/* ── RISK BAND BADGE ───────────────────────────────────────── */
.band-baixa    { color:#10B981; background:rgba(16,185,129,0.12); border-color:rgba(16,185,129,0.24); }
.band-moderada { color:#F59E0B; background:rgba(245,158,11,0.12); border-color:rgba(245,158,11,0.24); }
.band-elevada  { color:#EF4444; background:rgba(239,68,68,0.12);  border-color:rgba(239,68,68,0.24); }
.band-alta     { color:#DC2626; background:rgba(220,38,38,0.14);  border-color:rgba(220,38,38,0.30); }
.risk-badge {
    display:inline-flex; align-items:center; gap:.4rem;
    padding:.32rem .7rem; border-radius:999px;
    border:1px solid; font-size:.76rem; font-weight:800;
    text-transform:uppercase; letter-spacing:.06em;
}

/* ── MINI CALLOUT ──────────────────────────────────────────── */
.mini-callout {
    display:flex; gap:.9rem; align-items:flex-start;
    padding:.9rem 1rem; border-radius:var(--r-sm);
    border:1px solid var(--border);
    background: rgba(255,255,255,0.02);
    margin-bottom:.25rem;
    animation: fadeIn .4s ease;
}
.mini-callout-icon {
    flex:0 0 auto; width:1.85rem; height:1.85rem; border-radius:var(--r-xs);
    display:flex; align-items:center; justify-content:center;
    background: rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.20);
    color:#60A5FA; font-weight:900; font-size:.78rem;
}
.mini-callout-title { font-weight:800; font-size:.88rem; margin-bottom:.18rem; color:var(--text); }

/* ── INPUTS ────────────────────────────────────────────────── */
.stTextInput label, .stSelectbox label, .stMultiSelect label {
    font-size:.76rem !important; font-weight:700 !important;
    text-transform:uppercase; letter-spacing:.07em; color:var(--muted) !important;
}
/* Cabeçalhos HTML não tentam imitar contêineres interativos. Conteúdo dinâmico
   usa st.container(border=True), que mantém título e componente no mesmo bloco. */
.card:not(:has(.journey-grid)) {
    background:transparent; border:0; border-radius:0; box-shadow:none;
    padding:.25rem 0 0; margin:0;
}
.card:not(:has(.journey-grid)) .card-header {
    border-bottom:0; padding-bottom:.2rem; margin-bottom:.25rem;
}
[data-testid="stWidgetLabel"] { margin-bottom:.35rem; }
[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border-radius:var(--r-sm) !important;
    border:1px solid rgba(255,255,255,0.09) !important;
    color:var(--text) !important;
    min-height:44px;
    transition: border-color .15s, box-shadow .15s;
}
.stTextInput input:focus,
[data-baseweb="select"] > div:focus-within,
.stMultiSelect div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(59,130,246,0.40) !important;
    box-shadow: var(--ring) !important;
}
[data-baseweb="select"] svg { color:var(--muted) !important; }

/* ── BUTTON ────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
    width:100%; min-height:44px;
    background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
    color:#ffffff !important; border:none !important;
    border-radius:var(--r-sm) !important;
    font-weight:800 !important; font-size:.88rem !important;
    letter-spacing:.01em !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.30) !important;
    transition: transform .14s, box-shadow .14s, filter .14s !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 30px rgba(59,130,246,0.38) !important;
    filter: brightness(1.07) !important;
}

/* ── DATAFRAME ─────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.015);
    border:1px solid var(--border);
    border-radius:var(--r-sm);
    overflow:hidden;
}
[data-testid="stDataFrame"] thead tr th {
    background: rgba(255,255,255,0.04) !important;
    color:var(--muted) !important;
    font-size:.72rem !important; font-weight:700 !important;
    text-transform:uppercase; letter-spacing:.06em;
    border-bottom:1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: rgba(255,255,255,0.018); }
[data-testid="stDataFrame"] tbody tr:hover            { background: rgba(59,130,246,0.06) !important; }

/* ── TABS ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius:var(--r-sm); padding:.3rem;
    gap:.25rem;
}
.stTabs [data-baseweb="tab"] {
    font-weight:700; color:var(--muted);
    border-radius:var(--r-xs) !important;
    transition:color .14s, background .14s;
}
.stTabs [data-baseweb="tab"]:hover          { color:var(--accent-2); }
.stTabs [aria-selected="true"]              { color:var(--accent-2) !important; background: rgba(59,130,246,0.12) !important; }
.stTabs [data-baseweb="tab-highlight"]      { background:var(--accent) !important; }

/* ── ALERTS ────────────────────────────────────────────────── */
.stAlert {
    border-radius:var(--r-sm);
    border:1px solid var(--border);
    background: rgba(255,255,255,0.02);
    animation: fadeUp .5s ease-out;
}

/* ── SPINNER ───────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── TYPOGRAPHY ────────────────────────────────────────────── */
h3 {
    position:relative; font-size:1.45rem; font-weight:900; letter-spacing:-.035em;
    line-height:1.2; color:#FFFFFF; margin:.35rem 0 .15rem; padding-left:.85rem;
}
h3::before {
    content:''; position:absolute; left:0; top:.12rem; bottom:.12rem; width:3px;
    border-radius:99px; background:linear-gradient(180deg,var(--accent-2),var(--accent));
    box-shadow:0 0 16px rgba(59,130,246,.45);
}
[data-testid="stCaptionContainer"] p { color:var(--muted); font-size:.8rem; }
[data-testid="stMetric"]              { background:transparent; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], .stAlert { animation: fadeUp .5s ease-out; }

/* ── TOOLTIP KPI ───────────────────────────────────────────── */
.kpi-tooltip-wrap { position: relative; display: inline-block; }
.kpi-tooltip-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.1rem; height: 1.1rem; border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.12);
    color: var(--muted); font-size: .62rem; font-weight: 800;
    cursor: help; margin-left: .35rem; vertical-align: middle;
    transition: border-color .15s, color .15s;
}
.kpi-tooltip-icon:hover { border-color: var(--accent); color: var(--accent-2); }
.kpi-tooltip-box {
    visibility: hidden; opacity: 0;
    position: absolute; bottom: 130%; left: 50%;
    transform: translateX(-50%);
    min-width: 180px; max-width: 240px;
    background: rgba(15,22,40,0.98);
    border: 1px solid rgba(59,130,246,0.20);
    border-radius: var(--r-xs);
    padding: .55rem .7rem;
    font-size: .76rem; color: var(--muted); line-height: 1.5;
    box-shadow: 0 8px 24px rgba(0,0,0,0.60);
    z-index: 100;
    transition: opacity .15s, visibility .15s;
    pointer-events: none;
    white-space: normal;
    text-align: left;
}
.kpi-tooltip-wrap:hover .kpi-tooltip-box { visibility: visible; opacity: 1; }

/* ── LEVEL BADGE (inline HTML) ─────────────────────────────── */
.lvl-baixa    { color:#10B981; background:rgba(16,185,129,0.10); border:1px solid rgba(16,185,129,0.20); }
.lvl-moderada { color:#F59E0B; background:rgba(245,158,11,0.10); border:1px solid rgba(245,158,11,0.20); }
.lvl-elevada  { color:#EF4444; background:rgba(239,68,68,0.10);  border:1px solid rgba(239,68,68,0.20); }
.lvl-alta     { color:#DC2626; background:rgba(220,38,38,0.12);  border:1px solid rgba(220,38,38,0.25); }
.lvl-badge {
    display: inline-block; padding: .18rem .5rem; border-radius: 999px;
    font-size: .72rem; font-weight: 800; letter-spacing: .04em;
}

/* ── SEARCH HISTORY ────────────────────────────────────────── */
.hist-wrap {
    display: flex; align-items: center; gap: .5rem;
    flex-wrap: wrap; margin-bottom: .8rem;
}
.hist-label {
    font-size: .72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted);
}

/* ── COMPARE COLUMNS ───────────────────────────────────────── */
.compare-header {
    display: flex; align-items: center; gap: .6rem;
    padding: .55rem .95rem; border-radius: var(--r-sm);
    background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.18);
    font-size: .84rem; font-weight: 800; color: var(--accent-2);
    margin-bottom: .9rem;
}
.compare-inputs {
    display: grid; grid-template-columns: 1fr auto 1fr;
    gap: 0; align-items: start; margin-bottom: 1rem;
}
.compare-vs {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 0 1rem; margin-top: 1.6rem;
}
.compare-vs-badge {
    display: flex; align-items: center; justify-content: center;
    width: 2.4rem; height: 2.4rem; border-radius: 50%;
    background: rgba(59,130,246,0.12); border: 2px solid rgba(59,130,246,0.25);
    color: var(--accent-2); font-size: .8rem; font-weight: 900;
    box-shadow: 0 0 18px rgba(59,130,246,0.15);
}
.compare-vs-line {
    width: 1px; flex: 1; min-height: 40px;
    background: linear-gradient(to bottom, transparent, rgba(59,130,246,0.20), transparent);
}
.compare-side-label {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .35rem .75rem; border-radius: var(--r-xs);
    font-size: .75rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: .07em; margin-bottom: .65rem;
}
.compare-side-a { background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.22); color: #60A5FA; }
.compare-side-b { background: rgba(139,92,246,0.10); border: 1px solid rgba(139,92,246,0.22); color: #A78BFA; }
.compare-divider {
    width: 1px; background: linear-gradient(to bottom, transparent 0%, rgba(255,255,255,0.08) 20%, rgba(255,255,255,0.08) 80%, transparent 100%);
    margin: 0 .5rem; align-self: stretch;
}
.compare-result-wrap {
    padding: 1rem 1.1rem 1.1rem;
    border-radius: var(--r);
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-card);
}
.compare-result-a { border-top: 2px solid #3B82F6; }
.compare-result-b { border-top: 2px solid #8B5CF6; }

/* ── DIVIDER ───────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.05) !important; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border:1px solid rgba(148,163,184,.12) !important;
    border-radius:18px !important;
    background:linear-gradient(145deg,rgba(13,18,29,.96),rgba(9,13,22,.96)) !important;
    box-shadow:0 14px 42px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.025);
}
[data-testid="stVerticalBlockBorderWrapper"] > div { padding:1.2rem 1.3rem 1.25rem !important; }
.filter-heading { display:flex; align-items:center; gap:.7rem; margin:0 0 .15rem; }
.filter-heading-icon {
    display:grid; place-items:center; width:2rem; height:2rem; border-radius:9px;
    color:var(--accent-2); background:var(--accent-soft);
    border:1px solid rgba(59,130,246,.22); font-size:.82rem; font-weight:900;
}
.filter-heading-title { color:var(--text); font-size:.98rem; font-weight:850; }
.filter-heading-copy { color:var(--muted); font-size:.78rem; margin-top:.12rem; }
.section-heading { display:flex; align-items:center; gap:.75rem; padding:.05rem 0 .9rem; }
.section-heading .card-icon { width:2.15rem; height:2.15rem; }
.section-heading .section-title { margin:0; }
.section-heading .section-copy { margin:.18rem 0 0; }
[data-testid="stPlotlyChart"] { border-radius:12px; overflow:hidden; }
[data-testid="stDataFrame"] { min-height:260px; }
[data-testid="stDeckGlJsonChart"] { border:1px solid var(--border); border-radius:16px; overflow:hidden; }

/* ── RESPONSIVE ────────────────────────────────────────────── */
@media (max-width:980px) {
    .kpi-grid   { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .hero-kpis  { width:100%; justify-content:flex-start; }
}
@media (max-width:760px) {
    .journey-grid, .kpi-grid { grid-template-columns:1fr; }
    .hero h1  { font-size:2.1rem; }
    .hero     { padding:1.5rem 1.15rem 1.3rem; }
    .block-container { padding:1rem .9rem 3rem; }
    [data-testid="stHorizontalBlock"] { gap:.65rem; }
    .hero-kpi { min-width:0; flex:1 1 140px; }
    .nav-wrap { position:relative; top:0; overflow-x:auto; }
    div[role="radiogroup"] { flex-wrap:nowrap !important; min-width:max-content; }
    .panel, .card { padding:1.1rem; border-radius:15px; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { padding:1rem !important; }
}
</style>
"""


def inject_css() -> None:
    css = CSS_TEMPLATE
    replacements = {
        '__BG__':             THEME['bg'],
        '__BG_2__':           THEME['bg_2'],
        '__SURFACE__':        THEME['surface'],
        '__SURFACE_STRONG__': THEME['surface_strong'],
        '__SURFACE_SOFT__':   THEME['surface_soft'],
        '__TEXT__':           THEME['text'],
        '__MUTED__':          THEME['muted'],
        '__BORDER__':         THEME['border'],
        '__ACCENT__':         THEME['accent'],
        '__ACCENT_SOFT__':    THEME['accent_soft'],
        '__ACCENT_2__':       THEME['accent_2'],
        '__DANGER__':         THEME['danger'],
        '__OK__':             THEME['ok'],
        '__SHADOW__':         THEME['shadow'],
    }
    for k, v in replacements.items():
        css = css.replace(k, v)
    st.markdown(css, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        f"""
        <div class='hero'>
            <div class='hero-topline'>
                <span class='hero-eyebrow'>Análise de segurança viária</span>
                <div class='hero-meta'>
                    <span>Brasil e São Paulo</span>
                    <span>Dados de 2025 e 2026</span>
                </div>
            </div>
            <h1><em>{APP_TITLE}</em></h1>
            <div class='hero-sub'>{APP_SUBTITLE} {APP_DESCRIPTION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(icon: str, title: str, copy: str | None = None) -> None:
    copy_html = f"<div class='section-copy'>{copy}</div>" if copy else ''
    st.markdown(
        f"""
        <div class='section-heading'>
            <div class='card-icon'>{icon}</div>
            <div><div class='section-title'>{title}</div>{copy_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav() -> str:
    labels = [name for name, _ in PAGES]
    if 'page' not in st.session_state:
        st.session_state.page = 'Planejar viagem'
    page = st.radio(
        'Navegação',
        labels,
        index=labels.index(st.session_state.page),
        horizontal=True,
        label_visibility='collapsed',
        key='top_nav_radio',
    )
    st.session_state.page = page
    return page


def render_status_bar(total: int, source: str, counts: dict, uf_count: int) -> None:
    count_str = ' · '.join([f'{v:,} {k}' for k, v in counts.items()]) if counts else 'sem dados'
    st.markdown(
        f"""
        <div class='status-bar'>
            <span class='status-chip'>Registros <strong>{total:,}</strong></span>
            <span class='status-sep'>|</span>
            <span class='status-chip'>Base <strong>{source}</strong></span>
            <span class='status-sep'>|</span>
            <span class='status-chip'>Composição <strong>{count_str}</strong></span>
            <span class='status-sep'>|</span>
            <span class='status-chip'>Estados disponíveis <strong>{uf_count}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str | None = None, tooltip: str | None = None) -> None:
    tip_html = ''
    if tooltip:
        tip_html = (
            f"<span class='kpi-tooltip-wrap'>"
            f"<span class='kpi-tooltip-icon'>?</span>"
            f"<span class='kpi-tooltip-box'>{tooltip}</span>"
            f"</span>"
        )
    html = (
        f"<div class='kpi'>"
        f"<div class='kpi-label'>{label}{tip_html}</div>"
        f"<div class='kpi-value'>{value}</div>"
    )
    if help_text:
        html += f"<div class='kpi-help'>{help_text}</div>"
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_risk_badge(band: str) -> str:
    cls = f"band-{band.lower().replace('ç', 'c').replace('ã', 'a')}"
    return f"<span class='risk-badge {cls}'>{band}</span>"


def render_gauge(score: int, band: str) -> None:
    color_map = {
        'baixa':    '#10B981',
        'moderada': '#F59E0B',
        'elevada':  '#EF4444',
        'alta':     '#DC2626',
    }
    band_key = band.lower()
    bar_color = color_map.get(band_key, '#3B82F6')

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'font': {'size': 44, 'color': '#FFFFFF', 'family': 'Inter, sans-serif'},
            'suffix': '',
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': 'rgba(255,255,255,0.10)',
                'nticks': 6,
                'tickfont': {'color': '#5C6A80', 'size': 9},
            },
            'bar':   {'color': bar_color, 'thickness': 0.22},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25],   'color': 'rgba(16,185,129,0.08)'},
                {'range': [25, 50],  'color': 'rgba(245,158,11,0.08)'},
                {'range': [50, 75],  'color': 'rgba(239,68,68,0.08)'},
                {'range': [75, 100], 'color': 'rgba(220,38,38,0.12)'},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.75,
                'value': score,
            },
        },
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=18, r=18, t=14, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#E2E8F4', 'family': 'Inter, sans-serif'},
        annotations=[{
            'text': band.upper(),
            'x': 0.5, 'y': 0.20,
            'showarrow': False,
            'font': {'size': 11, 'color': bar_color, 'family': 'Inter, sans-serif'},
            'xanchor': 'center',
        }],
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def _render_filters_legacy(source_options: list[str], years: list[int]):
    st.markdown(
        """
        <div class='panel'>
            <div class='card-header'>
                <div class='card-icon'>▼</div>
                <div>
                    <div class='section-title'>Filtros</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.05, 1.55])
    source = c1.selectbox(
        'Base de dados',
        source_options,
        index=source_options.index(st.session_state.get('_source', source_options[0])),
        key='_source',
    )
    year_sel = c2.multiselect('Anos', years, default=st.session_state.get('_years', []), placeholder='Todos os anos', key='_years')
    st.markdown('</div>', unsafe_allow_html=True)
    return source, year_sel


def render_filters(
    source_options: list[str],
    years: list[int],
    ufs_by_source: dict[str, list[str]],
):
    """Render all global filters in one cohesive, responsive panel."""
    with st.container(border=True):
        st.markdown(
            """
            <div class='filter-heading'>
                <div class='filter-heading-icon'>⌁</div>
                <div>
                    <div class='filter-heading-title'>Refine sua consulta</div>
                    <div class='filter-heading-copy'>Combine base, período e estados para atualizar todo o painel.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([1.0, 1.25, 1.45], gap='medium')
        source = c1.selectbox(
            'Base de dados',
            source_options,
            index=source_options.index(st.session_state.get('_source', source_options[0])),
            key='_source',
        )
        selected_years = c2.multiselect(
            'Anos',
            years,
            default=st.session_state.get('_years', []),
            placeholder='Todos os anos',
            key='_years',
        )
        available_ufs = ufs_by_source.get(source, [])
        previous_ufs = [uf for uf in st.session_state.get('_ufs', []) if uf in available_ufs]
        if previous_ufs != st.session_state.get('_ufs', []):
            st.session_state['_ufs'] = previous_ufs
        selected_ufs = c3.multiselect(
            'Estados',
            available_ufs,
            default=previous_ufs,
            key='_ufs',
            placeholder='Todos os estados',
        )
    return source, selected_years, selected_ufs, available_ufs


def render_journey() -> None:
    st.markdown(
        """
        <div class='card'>
            <div class='card-header'>
                <div class='card-icon'>→</div>
                <div>
                    <div class='section-title'>Como usar o Radar</div>
                    <div class='section-copy' style='margin-bottom:0'>Siga os três passos para encontrar o melhor horário.</div>
                </div>
            </div>
            <div class='journey-grid'>
                <div class='journey-step'>
                    <span class='step-index'>1</span>
                    <strong>Escolha a base</strong>
                    <span class='muted'>Use os dados do Brasil, de São Paulo ou tudo junto.</span>
                </div>
                <div class='journey-step'>
                    <span class='step-index'>2</span>
                    <strong>Digite o destino</strong>
                    <span class='muted'>Com o destino e o dia da semana, o sistema mostra a dica principal.</span>
                </div>
                <div class='journey-step'>
                    <span class='step-index'>3</span>
                    <strong>Veja o melhor horário</strong>
                    <span class='muted'>Compare os horários e abra a rota no Google Maps.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_panel(title: str, summary: str, recommendation: str, bullets: list[str]) -> None:
    items = ''.join([f'<li>{b}</li>' for b in bullets])
    st.markdown(
        f"""
        <div class='result-hero'>
            <div class='result-tag'>Dica principal</div>
            <div class='result-title'>{title}</div>
            <div class='result-copy'>{summary}</div>
            <div class='subtle-line'></div>
            <div class='result-note'>{recommendation}</div>
            <ul class='result-list'>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_callout(icon: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class='mini-callout'>
            <div class='mini-callout-icon'>{icon}</div>
            <div>
                <div class='mini-callout-title'>{title}</div>
                <div class='muted'>{copy}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def human_hours(hours: list[int]) -> str:
    return ', '.join([f'{h:02d}h' for h in hours]) if hours else 'sem referência suficiente'


def apply_level_style(df_display: pd.DataFrame, level_col: str = 'Nível') -> object:
    def _color(val: str) -> str:
        palette = {
            'Baixa':    'background-color: rgba(16,185,129,0.12); color: #10B981;',
            'Moderada': 'background-color: rgba(245,158,11,0.12); color: #F59E0B;',
            'Elevada':  'background-color: rgba(239,68,68,0.12);  color: #EF4444;',
            'Alta':     'background-color: rgba(220,38,38,0.14);  color: #DC2626;',
        }
        return palette.get(str(val), '')

    if level_col in df_display.columns:
        try:
            return df_display.style.map(_color, subset=[level_col])
        except Exception:
            return df_display.style.applymap(_color, subset=[level_col])
    return df_display.style


def offwhite_bar_chart(
    df,
    x_col: str,
    y_col: str,
    *,
    title: str = '',
    y_title: str = 'Nível de atenção',
) -> go.Figure:
    frame = df.copy()
    y_values = [None if pd.isna(v) else float(v) for v in frame[y_col].tolist()]
    observed = frame.get('observed', pd.Series(True, index=frame.index))

    bar_colors = [
        'rgba(59,130,246,0.80)' if obs else 'rgba(255,255,255,0.08)'
        for obs in observed
    ]
    bar_lines = [
        'rgba(96,165,250,0.90)' if obs else 'rgba(255,255,255,0.12)'
        for obs in observed
    ]

    valid = [v for v in y_values if v is not None]
    avg   = sum(valid) / len(valid) if valid else None

    fig = go.Figure()

    fig.add_bar(
        x=frame[x_col].astype(str).tolist(),
        y=y_values,
        marker=dict(
            color=bar_colors,
            line=dict(color=bar_lines, width=1),
            cornerradius=4,
        ),
        text=[f'{v:.0f}' if v is not None else '' for v in y_values],
        textposition='outside',
        textfont=dict(size=9, color='#5C6A80'),
        hovertemplate='<b>%{x}</b><br>' + y_title + ': %{y:.1f}<extra></extra>',
        cliponaxis=False,
    )

    if avg is not None:
        x_labels = frame[x_col].astype(str).tolist()
        fig.add_scatter(
            x=x_labels,
            y=[avg] * len(x_labels),
            mode='lines',
            line=dict(color='rgba(96,165,250,0.45)', width=1.5, dash='dot'),
            hovertemplate=f'Média: {avg:.1f}<extra></extra>',
            name='Média',
        )

    fig.update_layout(
        height=330,
        margin=dict(l=8, r=8, t=46 if title else 16, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=title,
            x=0.01, xanchor='left',
            font=dict(size=14, color='#E2E8F4', family='Inter, sans-serif'),
        ),
        xaxis=dict(
            title='', showgrid=False,
            tickfont=dict(color='#5C6A80', size=11),
            linecolor='rgba(255,255,255,0.06)',
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color='#5C6A80', size=11)),
            tickfont=dict(color='#5C6A80', size=10),
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            rangemode='tozero',
        ),
        showlegend=False,
        font=dict(color='#E2E8F4', family='Inter, sans-serif'),
        bargap=0.22,
        hoverlabel=dict(
            bgcolor='rgba(11,15,24,0.97)',
            bordercolor='rgba(59,130,246,0.30)',
            font=dict(color='#E2E8F4'),
        ),
    )
    return fig


def trend_line_chart(trend_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if trend_df.empty:
        return fig

    anos = trend_df['ano'].astype(str).tolist()

    fig.add_scatter(
        x=anos, y=trend_df['acidentes'].tolist(),
        mode='lines+markers',
        name='Acidentes',
        line=dict(color='#3B82F6', width=2.5),
        marker=dict(size=7, color='#60A5FA', line=dict(color='#1E3A5F', width=1.5)),
        hovertemplate='<b>%{x}</b><br>Acidentes: %{y:,}<extra></extra>',
    )
    fig.add_scatter(
        x=anos, y=trend_df['mortos'].tolist(),
        mode='lines+markers',
        name='Mortos',
        line=dict(color='#EF4444', width=2),
        marker=dict(size=6, color='#EF4444'),
        hovertemplate='<b>%{x}</b><br>Mortos: %{y:,}<extra></extra>',
        yaxis='y2',
    )

    fig.update_layout(
        height=300,
        margin=dict(l=8, r=8, t=38, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text='Tendência por ano',
            x=0.01, xanchor='left',
            font=dict(size=14, color='#E2E8F4', family='Inter, sans-serif'),
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color='#5C6A80', size=11),
            linecolor='rgba(255,255,255,0.06)',
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='Acidentes', font=dict(color='#3B82F6', size=10)),
            tickfont=dict(color='#5C6A80', size=9),
            gridcolor='rgba(255,255,255,0.04)',
            zeroline=False,
        ),
        yaxis2=dict(
            title=dict(text='Mortos', font=dict(color='#EF4444', size=10)),
            tickfont=dict(color='#5C6A80', size=9),
            overlaying='y', side='right',
            zeroline=False, showgrid=False,
        ),
        legend=dict(
            x=0.01, y=0.99,
            font=dict(color='#5C6A80', size=10),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.06)',
            borderwidth=1,
        ),
        font=dict(color='#E2E8F4', family='Inter, sans-serif'),
        hoverlabel=dict(
            bgcolor='rgba(11,15,24,0.97)',
            bordercolor='rgba(59,130,246,0.30)',
            font=dict(color='#E2E8F4'),
        ),
    )
    return fig


def heatmap_chart(matrix_df: pd.DataFrame) -> go.Figure:
    if matrix_df.empty:
        return go.Figure()

    days  = matrix_df.index.tolist()
    hours = [str(c) for c in matrix_df.columns]
    z     = matrix_df.values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=hours,
        y=days,
        colorscale=[
            [0.0,  'rgba(16,185,129,0.20)'],
            [0.35, 'rgba(245,158,11,0.40)'],
            [0.65, 'rgba(239,68,68,0.55)'],
            [1.0,  'rgba(220,38,38,0.90)'],
        ],
        showscale=True,
        colorbar=dict(
            thickness=10,
            tickfont=dict(color='#5C6A80', size=9),
            title=dict(text='Score', font=dict(color='#5C6A80', size=10), side='right'),
            bgcolor='rgba(0,0,0,0)',
            outlinewidth=0,
        ),
        hovertemplate='<b>%{y} — %{x}h</b><br>Score: %{z:.0f}<extra></extra>',
        xgap=2,
        ygap=2,
    ))

    fig.update_layout(
        height=310,
        margin=dict(l=8, r=50, t=42, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text='Mapa de calor: dia × hora',
            x=0.01, xanchor='left',
            font=dict(size=14, color='#E2E8F4', family='Inter, sans-serif'),
        ),
        xaxis=dict(
            title='Hora',
            tickfont=dict(color='#5C6A80', size=10),
            linecolor='rgba(255,255,255,0.06)',
            showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(color='#5C6A80', size=10),
            showgrid=False,
            autorange='reversed',
        ),
        font=dict(color='#E2E8F4', family='Inter, sans-serif'),
        hoverlabel=dict(
            bgcolor='rgba(11,15,24,0.97)',
            bordercolor='rgba(59,130,246,0.30)',
            font=dict(color='#E2E8F4'),
        ),
    )
    return fig

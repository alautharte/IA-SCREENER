"""
app.py — Modelo IA Screener (USA & ARG)
Motor LINEST Walk-Forward Ortogonal · OLS Multitemporal · Golden Pocket · Multi-Usuario
Firma: LAUTHARTE · Zoom Estructural · v4.2
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN INICIAL
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Modelo IA Screener", page_icon="📊", layout="wide")

# ─────────────────────────────────────────────────────────────────
# MÓDULO DE AUTENTICACIÓN MULTI-USUARIO (LOGIN)
# ─────────────────────────────────────────────────────────────────
def check_password():
    def password_entered():
        user = st.session_state["username"]
        pwd = st.session_state["password"]
        if "passwords" in st.secrets:
            if user in st.secrets["passwords"] and st.secrets["passwords"][user] == pwd:
                st.session_state["password_correct"] = True
                st.session_state["logged_user"] = user
                del st.session_state["password"]
            else:
                st.session_state["password_correct"] = False
        else:
            if user == "admin" and pwd == "admin123":
                st.session_state["password_correct"] = True
                st.session_state["logged_user"] = user
            else:
                st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("## 🔐 Acceso Restringido")
        st.markdown("Plataforma Cuantitativa Institucional. Por favor, identifíquese.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Usuario", key="username")
            st.text_input("Contraseña", type="password", key="password")
            st.button("Ingresar al Sistema", on_click=password_entered, use_container_width=True)
            if st.session_state.get("password_correct") is False:
                st.error("❌ Usuario o contraseña incorrectos.")
        st.stop()

check_password()
usuario_actual = st.session_state["logged_user"]

# ─────────────────────────────────────────────────────────────────
# PARÁMETROS DEL MODELO
# ─────────────────────────────────────────────────────────────────
LAG_INICIAL   = 51
VENTANA_TRAIN = 252
BLIND_SPOT    = 20
F_UMBRAL      = 2.6
R2_MIN        = 0.01

VIX_CONTEXTOS = {
    "EUFORIA":       (0,  15,  1.00, "🟢", "#34d399"),
    "OPTIMISMO":     (15, 24,  1.05, "🔵", "#60a5fa"),
    "INCERTIDUMBRE": (24, 32,  0.90, "🟡", "#facc15"),
    "PANICO":        (32, 999, 0.75, "🔴", "#f87171"),
}

FEATS_M1 = ["rsi",      "atr_pct", "fuerza_rel", "ret_1d", "ret_3d"]
FEATS_M2 = ["macd_var", "atr_pct", "fuerza_rel", "ret_5d", "gap_oc"]
FEATS_M3 = ["mm50_var", "mm10_vs_mm50", "vol_var20", "ret_3d", "gap_oc"]

# ─────────────────────────────────────────────────────────────────
# UNIVERSOS DINÁMICOS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def cargar_universo_usa():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        sp = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', storage_options=headers)[0]
        u = [t.replace('.', '-') for t in sp['Symbol'].tolist()]
        return sorted(list(set(u)))
    except Exception:
        return sorted(["AAPL","NVDA","TSLA","META","AMZN","MSFT","GOOGL","JPM","V","WMT","PLTR","AMD","NFLX"])

@st.cache_data(ttl=86400, show_spinner=False)
def cargar_universo_arg():
    return sorted(["ALUA.BA","BBAR.BA","BMA.BA","BYMA.BA","CEPU.BA","COME.BA","CRES.BA","EDN.BA","GGAL.BA","LOMA.BA","MIRG.BA","PAMP.BA","SUPV.BA","TECO2.BA","TGNO4.BA","TGSU2.BA","TRAN.BA","TXAR.BA","VALO.BA","YPFD.BA","AGRO.BA","AUSO.BA","BHIP.BA","BOLT.BA","BPAT.BA","CADO.BA","CAPX.BA","CECO2.BA","CELU.BA","CGPA2.BA","CTIO.BA","CVH.BA","DGCU2.BA","FERR.BA","FIPL.BA","GAMI.BA","GARO.BA","GBAN.BA","GCLA.BA","GRIM.BA","HAVH.BA","INVJ.BA","IRSA.BA","LEDE.BA","LONG.BA","METR.BA","MOLI.BA","MORI.BA","OEST.BA","PATA.BA","RICH.BA","RIGO.BA","SAMI.BA","SEMI.BA"])

class ErrorLogger:
    def __init__(self): self.logs = []
    def add(self, ticker, motivo, detalle=""): self.logs.append({"Ticker": ticker, "Motivo": motivo, "Detalle": detalle})
    def to_df(self): return pd.DataFrame(self.logs) if self.logs else pd.DataFrame(columns=["Ticker", "Motivo", "Detalle"])

# ─────────────────────────────────────────────────────────────────
# DESCARGAS E INDICADORES
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def descargar(ticker, years):
    try:
        df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
        if df is None or df.empty or len(df) < 100: return None, "Error Datos", "Vacio/Corto"
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].copy(), None, None
    except Exception as e: return None, "Error API", str(e)[:40]

@st.cache_data(ttl=900, show_spinner=False)
def descargar_vix(years):
    try:
        df = yf.Ticker("^VIX").history(period=f"{years}y")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df["Close"].ffill()
    except Exception: return pd.Series(dtype=float)

@st.cache_data(ttl=900, show_spinner=False)
def descargar_benchmark(mercado, years):
    b = "SPY" if "Unidos" in mercado else "GGAL.BA"
    try:
        df = yf.Ticker(b).history(period=f"{years}y", auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df["Close"].ffill()
    except Exception: return pd.Series(dtype=float)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def calcular_indicadores(df, bench_serie, horizonte=20):
    d = df.copy()
    c, v, o = d["Close"], d["Volume"], d["Open"]
    d["fuerza_rel"] = (c.pct_change(20) - bench_serie.reindex(d.index, method="ffill").pct_change(20)) if not bench_serie.empty else 0.0
    d["ema12"], d["ema26"] = ema(c, 12), ema(c, 26)
    d["macd"] = d["ema12"] - d["ema26"]
    d["macd_sig"] = ema(d["macd"], 9)
    d["macd_var"] = (d["macd"] - d["macd_sig"]) / c.replace(0, np.nan)
    
    diff = c.diff()
    g = diff.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    p = (-diff).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - (100 / (1 + g/p.replace(0, np.nan)))
    
    d["mm10"], d["mm50"] = c.rolling(10).mean(), c.rolling(50).mean()
    d["mm50_var"] = (c - d["mm50"]) / d["mm50"].replace(0, np.nan)
    d["mm10_vs_mm50"] = (d["mm10"] - d["mm50"]) / d["mm50"].replace(0, np.nan)
    d["vol_var20"] = (v - v.rolling(20).mean()) / v.rolling(20).mean()
    
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_upper"], d["bb_mid"], d["bb_lower"] = m + 2*s, m, m - 2*s
    
    if "High" in d.columns:
        tr = pd.concat([(d["High"]-d["Low"]), (d["High"]-c.shift()).abs(), (d["Low"]-c.shift()).abs()], axis=1).max(axis=1)
        d["atr_pct"] = tr.ewm(alpha=1/14, adjust=False).mean() / c.replace(0, np.nan)
        lo, hi = d["Low"].rolling(14).min(), d["High"].rolling(14).max()
        d["stoch_k"] = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
        d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    
    d["ret_1d"], d["ret_3d"], d["ret_5d"] = c.pct_change(1), c.pct_change(3), c.pct_change(5)
    d["gap_oc"] = (c - o) / o.replace(0, np.nan)
    d["retorno_target"] = c.shift(-horizonte) / c - 1
    d["retorno_diario"] = c.pct_change()
    return d

def detectores_heuristicos(df):
    c, v, h, l = df["Close"], df["Volume"], df["High"], df["Low"]
    bk = bool(c.iloc[-1] >= c.rolling(50).max().iloc[-2]) if len(c)>50 else False
    ins = bool(v.iloc[-1] > v.rolling(20).mean().iloc[-1] * 2) if len(v)>20 else False
    exp = bool(c.pct_change(20).iloc[-1] > 0.15) if len(c)>20 else False
    fib = False
    if len(df)>60:
        mx, mn = h.iloc[-60:].max(), l.iloc[-60:].min()
        r = mx - mn
        if r>0:
            f50, f618 = mx - (r*0.5), mx - (r*0.618)
            if min(f50, f618)*0.985 <= c.iloc[-1] <= max(f50, f618)*1.015: fib = True
    return bk, ins, exp, fib

def contexto_vix(vix):
    for n, (lo, hi, f, i, c) in VIX_CONTEXTOS.items():
        if lo <= vix < hi: return n, f, i, c
    return "OPTIMISMO", 1.05, "🔵", "#60a5fa"

# ─────────────────────────────────────────────────────────────────
# MOTOR DE CÁLCULO OLS
# ─────────────────────────────────────────────────────────────────
def _normalizar(X_tr, x_pr):
    mu, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    std[std == 0] = 1.0
    return (X_tr - mu) / std, (x_pr - mu) / std

def _walk_forward_features(d, feats, y_f, N, ini_wf):
    X_f, k = d[feats].values, len(feats)
    preds, pesos = np.full(N, np.nan), np.full(N, 0.0)
    for i in range(ini_wf, N):
        fn, in_ = i - BLIND_SPOT, max(LAG_INICIAL, i - BLIND_SPOT - VENTANA_TRAIN)
        Xt_v, yt_v = X_f[in_:fn], y_f[in_:fn]
        mask = np.all(np.isfinite(Xt_v), axis=1) & np.isfinite(yt_v)
        if mask.sum() < 50 or not np.all(np.isfinite(X_f[i])): continue
        Xn, xn = _normalizar(Xt_v[mask], X_f[i])
        X_mat, x_vec = np.column_stack([Xn, np.ones(mask.sum())]), np.append(xn, 1.0)
        cfs, _, _, _ = np.linalg.lstsq(X_mat, yt_v[mask], rcond=None)
        yp = X_mat @ cfs
        sst = np.sum((yt_v[mask] - yt_v[mask].mean())**2)
        if sst <= 0: continue
        r2c = 1.0 - np.sum((yt_v[mask] - yp)**2) / sst
        r2a = 1.0 - (1.0 - r2c) * (mask.sum()-1) / (mask.sum()-k-1)
        if r2c<=0 or (r2c/k)/((1.0-r2c)/(mask.sum()-k-1)) < F_UMBRAL or r2a < R2_MIN: continue
        preds[i], pesos[i] = float(x_vec @ cfs), r2a
    return float(preds[-1]) if np.isfinite(preds[-1]) else 0.0, preds, float(pesos[-1]), pesos

def ejecutar_modelo(d, h):
    vacio = dict(consenso=0, r2_prom=0, pred_rsi=0, pred_macd=0, pred_medias=0, r2_rsi=0, r2_macd=0, r2_medias=0)
    N, ini = len(d), LAG_INICIAL + VENTANA_TRAIN + BLIND_SPOT
    if N < ini + 10: return vacio, d
    yf = d["retorno_target"].values
    p1, h1, w1, pw1 = _walk_forward_features(d, FEATS_M1, yf, N, ini)
    p2, h2, w2, pw2 = _walk_forward_features(d, FEATS_M2, yf, N, ini)
    p3, h3, w3, pw3 = _walk_forward_features(d, FEATS_M3, yf, N, ini)
    sum_w = w1+w2+w3
    res = dict(consenso=round((p1*w1+p2*w2+p3*w3)/sum_w, 6) if sum_w>0 else 0.0, 
               r2_prom=round(sum_w/sum(1 for w in [w1,w2,w3] if w>0), 4) if sum_w>0 else 0.0,
               pred_rsi=p1, pred_macd=p2, pred_medias=p3, r2_rsi=w1, r2_macd=w2, r2_medias=w3)
    p_h = pw1 + pw2 + pw3
    with np.errstate(divide="ignore", invalid="ignore"):
        d["consenso_raw"] = np.where(p_h > 0, (np.nan_to_num(h1)*pw1 + np.nan_to_num(h2)*pw2 + np.nan_to_num(h3)*pw3) / p_h, np.nan)
    return res, d

def ejecutar_modelo_multitemporal(d, vix_s, log, tk):
    N, ini = len(d), LAG_INICIAL + VENTANA_TRAIN + BLIND_SPOT
    if N < ini + 10: return None
    c = d["Close"]
    Ym = np.column_stack([(c.shift(-10)/c-1).values, (c.shift(-20)/c-1).values, (c.shift(-30)/c-1).values])
    def wf(f):
        Xf, k = d[f].values, len(f)
        pm, wm = np.full((N, 3), np.nan), np.zeros((N, 3))
        for i in range(ini, N):
            fn, st = i - BLIND_SPOT, max(LAG_INICIAL, i - BLIND_SPOT - VENTANA_TRAIN)
            Xv, Yv = Xf[st:fn], Ym[st:fn]
            m = np.all(np.isfinite(Xv), 1) & np.all(np.isfinite(Yv), 1)
            if m.sum()<50 or not np.all(np.isfinite(Xf[i])): continue
            Xn, xn = _normalizar(Xv[m], Xf[i])
            Xmat, xvec = np.column_stack([Xn, np.ones(m.sum())]), np.append(xn, 1.0)
            try: cfs, _, _, _ = np.linalg.lstsq(Xmat, Yv[m], rcond=None)
            except: continue
            for j in range(3):
                yp, ytj = Xmat @ cfs[:, j], Yv[m][:, j]
                sst = np.sum((ytj - ytj.mean())**2)
                if sst<=0: continue
                r2c = 1.0 - np.sum((ytj - yp)**2)/sst
                r2a = 1.0 - (1.0-r2c)*(m.sum()-1)/(m.sum()-k-1)
                if r2c>0 and (r2c/k)/((1.0-r2c)/(m.sum()-k-1))>=F_UMBRAL and r2a>=R2_MIN:
                    pm[i, j], wm[i, j] = float(xvec @ cfs[:, j]), r2a
        return pm, wm
    p1, w1 = wf(FEATS_M1); p2, w2 = wf(FEATS_M2); p3, w3 = wf(FEATS_M3)
    vh = float(d["vix"].iloc[-1]) if "vix" in d.columns else 18.0
    _, cf, _, _ = contexto_vix(vh)
    fz, r2 = [], []
    for j in range(3):
        ts = w1[-1,j] + w2[-1,j] + w3[-1,j]
        fz.append(((np.nan_to_num(p1[-1,j])*w1[-1,j] + np.nan_to_num(p2[-1,j])*w2[-1,j] + np.nan_to_num(p3[-1,j])*w3[-1,j])/ts*cf) if ts>0 else 0.0)
        r2.append((ts/sum(1 for w in [w1[-1,j],w2[-1,j],w3[-1,j]] if w>0)) if ts>0 else 0.0)
    if max(r2) < R2_MIN: return None
    vc, vv, fm = sum(1 for f in fz if f>0.02), sum(1 for f in fz if f<-0.02), float(np.mean(fz))
    s = "COMPRA FUERTE (3/3)" if vc==3 else ("COMPRAR" if vc>=2 and fm>0.02 else ("VENTA FUERTE (3/3)" if vv==3 else ("VENDER" if vv>=2 and fm<-0.02 else "ESPERAR / MIXTO")))
    return {"señal": s, "f_10d": round(fz[0], 4), "f_20d": round(fz[1], 4), "f_30d": round(fz[2], 4), "fuerza_media": round(fm, 4), "r2_medio": round(float(np.mean(r2)), 4)}

def calcular_auditoria_mtm(d, vix_s, h):
    da = d.copy()
    da["vix"] = vix_s.reindex(da.index, method="ffill")
    conds = [da["vix"]<15, (da["vix"]>=15)&(da["vix"]<24), (da["vix"]>=24)&(da["vix"]<32), da["vix"]>=32]
    da["consenso_final"] = da["consenso_raw"] * np.select(conds, [1.0, 1.05, 0.9, 0.75], 1.0)
    da["señal_h"] = np.where(da["consenso_final"]>0.02, "COMPRAR", np.where(da["consenso_final"]<-0.02, "VENDER", "ESPERAR"))
    tr = da[da["señal_h"]!="ESPERAR"].dropna(subset=["retorno_target"]).copy()
    if not tr.empty:
        tr["resultado"] = np.where(((tr["señal_h"]=="COMPRAR") & (tr["retorno_target"]>0)) | ((tr["señal_h"]=="VENDER") & (tr["retorno_target"]<0)), "✅ ACIERTO", "❌ FALLO")
    rd = (da["señal_h"].map({"COMPRAR":1, "VENDER":-1, "ESPERAR":0}).replace(0, np.nan).ffill(limit=h-1).fillna(0).shift(1) * da["retorno_diario"]).dropna()
    met = {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "win_rate": 0.0}
    if len(rd)>5:
        met["sharpe"] = float(np.sqrt(252)*rd.mean()/rd.std()) if rd.std()!=0 else 0.0
        da["equity_curve"] = (1+rd).cumprod()
        met["max_dd"] = float(((da["equity_curve"]-da["equity_curve"].cummax())/da["equity_curve"].cummax()).min())
        if not tr.empty: met["win_rate"] = float((tr["resultado"]=="✅ ACIERTO").sum()/len(tr))
    return tr, da, met

# ─────────────────────────────────────────────────────────────────
# UI - PANEL PRINCIPAL
# ─────────────────────────────────────────────────────────────────
for k in ["df_rank", "rank_mercado", "rank_anios"]:
    if k not in st.session_state: st.session_state[k] = None

with st.sidebar:
    st.markdown(f"👤 **Usuario Activo:** `{usuario_actual}`")
    if st.button("Cerrar Sesión"): st.session_state["password_correct"] = False; st.rerun()
    st.markdown("---")
    mercado = st.radio("Mercado", ["🇺🇸 Estados Unidos", "🇦🇷 Argentina (Merval)"])
    lista = cargar_universo_usa() if "Unidos" in mercado else cargar_universo_arg()
    ticker = st.selectbox("Activo Individual", lista)
    anios = st.slider("Años historia", 2, 10, 3)
    horizonte = st.slider("Horizonte (días)", 5, 60, 20, step=5)
    st.markdown("---")
    show_bb, show_vol = st.checkbox("Bandas Bollinger", True), st.checkbox("Volumen", True)
    show_stoch, show_atr = st.checkbox("Estocástico %K/%D", False), st.checkbox("ATR (%)", False)
    if st.button("🔄 Limpiar caché"): st.cache_data.clear(); st.rerun()

st.markdown("# 📊 Modelo IA Screener")
st.markdown(f"`{ticker}` · {mercado} · Objetivo: **{horizonte} días**")

with st.spinner(f"Analizando {ticker}..."):
    df_raw, em, ed = descargar(ticker, anios)
    vix_s = descargar_vix(anios)
    bench_s = descargar_benchmark(mercado, anios)

if df_raw is None: st.error(f"⚠️ {em}"); st.stop()

d = calcular_indicadores(df_raw, bench_s, horizonte)
d["vix"] = vix_s.reindex(d.index, method="ffill")
mod_res, d = ejecutar_modelo(d, horizonte)
bk, ins, exp, fib = detectores_heuristicos(df_raw)

h = d.iloc[-1]
vh = float(h.get("vix", 18.0) or 18.0)
cn, cf, ci, _ = contexto_vix(vh)
cons_adj = mod_res["consenso"] * cf
s_h = "COMPRAR" if cons_adj > 0.02 else ("VENDER" if cons_adj < -0.02 else "ESPERAR")
sc = {"COMPRAR":"#34d399", "VENDER":"#f87171", "ESPERAR":"#facc15"}[s_h]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(f"<div style='background:#1e293b;border-radius:8px;padding:12px;text-align:center'><div style='font-size:11px;color:#64748b'>Señal {horizonte}d</div><div style='font-size:1.7rem;font-weight:700;color:{sc}'>{s_h}</div><div style='font-size:12px;color:#64748b'>{cons_adj*100:+.2f}%</div></div>", unsafe_allow_html=True)
c2.metric("Precio", f"${h['Close']:.2f}"); c3.metric(f"VIX {ci}", f"{vh:.2f}", cn); c4.metric("RSI-14", f"{h['rsi']:.1f}"); c5.metric("MACD", f"{h['macd']:.4f}"); c6.metric("R² Prom", f"{mod_res['r2_prom']*100:.1f}%")

if any([bk, ins, exp, fib]):
    tags = [t for t, c in zip(["🚀 Breakout", "🏦 Inst Acc", "🔥 Momentum", "📐 Fibo"], [bk, ins, exp, fib]) if c]
    st.markdown(f"**Banderas:** `{'` · `'.join(tags)}`")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Gráfico", "🧠 Modelo", "🕵️ Auditoría", "📋 Datos", "🏆 Ranking", "💼 Cartera"])

with tab1:
    rows_n = 1 + sum([show_vol, show_stoch, show_atr])
    h_rows = [0.55] + [0.15] * (rows_n - 1)
    subs = ["Precio"] + (["Volumen"] if show_vol else []) + (["Estocástico"] if show_stoch else []) + (["ATR %"] if show_atr else [])
    fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=h_rows, subplot_titles=subs)
    
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Precio", increasing_line_color="#34d399", decreasing_line_color="#f87171", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["mm10"], name="MM10", line=dict(color="#facc15", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["mm50"], name="MM50", line=dict(color="#f87171", width=1.5, dash="dot")), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], name="BB+", line=dict(color="rgba(148,163,184,0.3)")), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], name="BB-", fill="tonexty", fillcolor="rgba(148,163,184,0.05)"), row=1, col=1)
    if cons_adj != 0.0 and mod_res["r2_prom"]>=R2_MIN:
        obj = h["Close"]*(1+cons_adj)
        fig.add_hline(y=obj, line_dash="dash", line_color=sc, annotation_text=f"Objetivo: ${obj:.2f}", row=1, col=1)

    curr = 2
    if show_vol:
        clrs = ["#34d399" if cl>=op else "#f87171" for cl, op in zip(d["Close"], d["Open"])]
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=clrs, showlegend=False), row=curr, col=1)
        fig.update_yaxes(fixedrange=True, row=curr, col=1); curr += 1
    if show_stoch and "stoch_k" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["stoch_k"], name="%K", line=dict(color="#60a5fa")), row=curr, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["stoch_d"], name="%D", line=dict(color="#f472b6", dash="dot")), row=curr, col=1)
        fig.update_yaxes(fixedrange=True, row=curr, col=1); curr += 1
    if show_atr and "atr_pct" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["atr_pct"]*100, name="ATR %", fill="tozeroy"), row=curr, col=1)
        fig.update_yaxes(fixedrange=True, row=curr, col=1)

    fig.update_layout(height=600 + (rows_n-1)*130, xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🧠 Resultados LINEST")
    c = st.columns(3)
    for i, (n, p, r) in enumerate(zip(["M1", "M2", "M3"], [mod_res["pred_rsi"], mod_res["pred_macd"], mod_res["pred_medias"]], [mod_res["r2_rsi"], mod_res["r2_macd"], mod_res["r2_medias"]])):
        with c[i]:
            st.metric(n, f"{p*100:+.2f}%", f"R² {r*100:.1f}%")

with tab3:
    tr, da, met = calcular_auditoria_mtm(d, vix_s, horizonte)
    if not tr.empty:
        st.metric("Win Rate", f"{met['win_rate']*100:.1f}%")
        st.dataframe(tr.sort_index(ascending=False), use_container_width=True)

with tab5:
    if st.button("🚀 Escanear Mercado", type="primary"):
        barra, res = st.progress(0, "Iniciando..."), []
        for i, a in enumerate(lista):
            barra.progress((i+1)/len(lista), f"Escaneando {a}")
            dfa, _, _ = descargar(a, anios)
            if dfa is not None:
                db = calcular_indicadores(dfa, bench_s, 20)
                db["vix"] = vix_s.reindex(db.index, method="ffill")
                mr = ejecutar_modelo_multitemporal(db, vix_s, None, a)
                if mr: res.append(mr | {"Activo": a, "Precio": dfa["Close"].iloc[-1]})
        if res: st.session_state["df_rank"] = pd.DataFrame(res).sort_values("fuerza_media", ascending=False)
    if st.session_state["df_rank"] is not None:
        st.dataframe(st.session_state["df_rank"], use_container_width=True, hide_index=True)

with tab6:
    st.markdown("### 💼 Mi Cartera")
    from streamlit_gsheets import GSheetsConnection
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_gs = conn.read(worksheet="Sheet1").dropna(how="all")
    if "Usuario" not in df_gs.columns: df_gs["Usuario"] = "admin"
    df_u = df_gs[df_gs["Usuario"] == usuario_actual].copy()
    
    with st.expander("➕ Operación"):
        with st.form("f_c"):
            t_c = st.selectbox("Ticker", sorted(list(set(cargar_universo_usa()+cargar_universo_arg()))))
            p_c = st.number_input("Precio", min_value=0.01)
            f_c = st.date_input("Fecha")
            h_c = st.selectbox("Horiz", [10, 20, 30])
            if st.form_submit_button("Cargar"):
                nf = pd.DataFrame([{"Usuario":usuario_actual,"Activo":t_c,"Fecha_Compra":f_c.strftime("%Y-%m-%d"),"Precio_Compra":p_c,"Horizonte_Dias":h_c,"Estado":"ABIERTA"}])
                conn.update(worksheet="Sheet1", data=pd.concat([df_gs, nf], ignore_index=True))
                st.success("Cargado"); st.rerun()

    if not df_u.empty:
        st.markdown("#### Posiciones Abiertas")
        st.dataframe(df_u[df_u["Estado"]=="ABIERTA"], use_container_width=True, hide_index=True)
        with st.form("f_v"):
            tk_v = st.selectbox("Vender", df_u[df_u["Estado"]=="ABIERTA"]["Activo"].tolist() if not df_u[df_u["Estado"]=="ABIERTA"].empty else [])
            pr_v = st.number_input("Precio Venta", min_value=0.01)
            if st.form_submit_button("Cerrar Posición") and tk_v:
                idx = df_gs[(df_gs["Usuario"]==usuario_actual) & (df_gs["Activo"]==tk_v) & (df_gs["Estado"]=="ABIERTA")].index[0]
                df_gs.at[idx, "Estado"], df_gs.at[idx, "Precio_Cierre"], df_gs.at[idx, "Fecha_Cierre"] = "CERRADA", pr_v, datetime.today().strftime("%Y-%m-%d")
                conn.update(worksheet="Sheet1", data=df_gs); st.success("Cerrada"); st.rerun()

# ─────────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("**Modelo IA Screener v4.2** | Desarrollado por: **LAUTHARTE**")
st.caption("⚠️ **Aviso Legal:** Análisis cuantitativo educativo. NO constituye asesoramiento financiero. Los resultados históricos no garantizan rendimientos futuros.")

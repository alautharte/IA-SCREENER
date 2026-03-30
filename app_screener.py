"""
app.py — Modelo IA Screener (USA & ARG)
Motor LINEST Walk-Forward Ortogonal · OLS Multitemporal · Golden Pocket · Multi-Usuario (Opción B)
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
        
        # Validar contra los Secrets de Streamlit
        if "passwords" in st.secrets:
            if user in st.secrets["passwords"] and st.secrets["passwords"][user] == pwd:
                st.session_state["password_correct"] = True
                st.session_state["logged_user"] = user
                del st.session_state["password"]  # Eliminar contraseña de memoria
            else:
                st.session_state["password_correct"] = False
        else:
            # Fallback de emergencia por si olvidaste configurar los secrets
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
                st.error("❌ Usuario o contraseña incorrectos. Verifique sus credenciales.")
        st.stop() # Bloquea la ejecución del resto del código si no está logueado

check_password()
usuario_actual = st.session_state["logged_user"]

# ─────────────────────────────────────────────────────────────────
# PARÁMETROS DEL MODELO
# ─────────────────────────────────────────────────────────────────
LAG_INICIAL   = 51
VENTANA_TRAIN = 252
BLIND_SPOT    = 20    # ESTRICTAMENTE FIJO PARA EVITAR DATA LEAKAGE
F_UMBRAL      = 2.6
R2_MIN        = 0.01
HORIZONTES_RANK = [10, 20, 30]

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
    sp500, ndx = [], []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        sp_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', storage_options=headers)[0]
        sp500 = [t.replace('.', '-') for t in sp_table['Symbol'].tolist()]
    except Exception: pass 
    try:
        ndx_table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100', storage_options=headers)[4]
        ndx = [t.replace('.', '-') for t in ndx_table['Ticker'].tolist()]
    except Exception: pass
    universo = sorted(list(set(sp500 + ndx)))
    if not universo:
        return sorted(["AAPL","NVDA","TSLA","META","AMZN","MSFT","GOOGL","JPM","V","WMT"])
    return universo

@st.cache_data(ttl=86400, show_spinner=False)
def cargar_universo_arg():
    return sorted([
        "ALUA.BA","BBAR.BA","BMA.BA","BYMA.BA","CEPU.BA","COME.BA","CRES.BA",
        "EDN.BA","GGAL.BA","LOMA.BA","MIRG.BA","PAMP.BA","SUPV.BA","TECO2.BA",
        "TGNO4.BA","TGSU2.BA","TRAN.BA","TXAR.BA","VALO.BA","YPFD.BA",
        "AGRO.BA","AUSO.BA","BHIP.BA","BOLT.BA","BPAT.BA","CADO.BA","CAPX.BA",
        "CECO2.BA","CELU.BA","CGPA2.BA","CTIO.BA","CVH.BA","DGCU2.BA",
        "FERR.BA","FIPL.BA","GAMI.BA","GARO.BA","GBAN.BA","GCLA.BA","GRIM.BA",
        "HAVH.BA","INVJ.BA","IRSA.BA","LEDE.BA","LONG.BA","METR.BA","MOLI.BA",
        "MORI.BA","OEST.BA","PATA.BA","RICH.BA","RIGO.BA","SAMI.BA","SEMI.BA"
    ])

class ErrorLogger:
    def __init__(self): self.logs = []
    def add(self, ticker, motivo, detalle=""): self.logs.append({"Ticker": ticker, "Motivo": motivo, "Detalle": detalle})
    def to_df(self): return pd.DataFrame(self.logs) if self.logs else pd.DataFrame(columns=["Ticker", "Motivo", "Detalle"])

# ─────────────────────────────────────────────────────────────────
# DESCARGAS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def descargar(ticker, years):
    try:
        df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
        if df is None or df.empty: return None, "Sin datos", "Respuesta vacía"
        if len(df) < 100: return None, "Datos insuficientes", f"n={len(df)} < 100"
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].copy(), None, None
    except Exception as e: return None, "Error API", str(e)[:60]

@st.cache_data(ttl=900, show_spinner=False)
def descargar_vix(years):
    try:
        df = yf.Ticker("^VIX").history(period=f"{years}y")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df["Close"].ffill()
    except Exception: return pd.Series(dtype=float)

@st.cache_data(ttl=900, show_spinner=False)
def descargar_benchmark(mercado, years):
    bench = "SPY" if "Unidos" in mercado else "GGAL.BA"
    try:
        df = yf.Ticker(bench).history(period=f"{years}y", auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df["Close"].ffill()
    except Exception: return pd.Series(dtype=float)

# ─────────────────────────────────────────────────────────────────
# INDICADORES & DETECTORES
# ─────────────────────────────────────────────────────────────────
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi_wilder(precio, periodo=14):
    d = precio.diff()
    gan = d.clip(lower=0).ewm(alpha=1/periodo, adjust=False).mean()
    per = (-d).clip(lower=0).ewm(alpha=1/periodo, adjust=False).mean()
    rs = gan / per.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger(precio, w=20, std=2.0):
    m = precio.rolling(w).mean()
    s = precio.rolling(w).std()
    return m + std*s, m, m - std*s

def atr_calc(high, low, close, w=14):
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/w, adjust=False).mean()

def stochastic(high, low, close, k=14, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    pct_k = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    return pct_k, pct_k.rolling(d).mean()

def calcular_indicadores(df, bench_serie, horizonte=20):
    d = df.copy()
    c, v, o = d["Close"], d["Volume"], d["Open"]
    d["fuerza_rel"] = (c.pct_change(20) - bench_serie.reindex(d.index, method="ffill").pct_change(20)) if not bench_serie.empty else 0.0

    d["ema12"], d["ema26"] = ema(c, 12), ema(c, 26)
    d["macd"] = d["ema12"] - d["ema26"]
    d["macd_sig"] = ema(d["macd"], 9)
    d["macd_var"] = (d["macd"] - d["macd_sig"]) / c.replace(0, np.nan)
    d["rsi"] = rsi_wilder(c)
    d["mm10"], d["mm50"] = c.rolling(10).mean(), c.rolling(50).mean()
    d["mm50_var"] = (c - d["mm50"]) / d["mm50"].replace(0, np.nan)
    d["mm10_vs_mm50"] = (d["mm10"] - d["mm50"]) / d["mm50"].replace(0, np.nan)
    d["vol_var20"] = (v - v.rolling(20).mean()) / v.rolling(20).mean()
    d["bb_upper"], d["bb_mid"], d["bb_lower"] = bollinger(c)
    d["bb_pct"] = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)
    d["volatilidad"] = c.pct_change().rolling(20).std() * np.sqrt(252)

    if all(col in d.columns for col in ["High", "Low"]):
        d["atr"] = atr_calc(d["High"], d["Low"], c)
        d["atr_pct"] = d["atr"] / c.replace(0, np.nan)
        d["stoch_k"], d["stoch_d"] = stochastic(d["High"], d["Low"], c)
    else:
        d["atr_pct"] = np.nan

    d["ret_1d"], d["ret_3d"], d["ret_5d"] = c.pct_change(1), c.pct_change(3), c.pct_change(5)
    d["gap_oc"] = (c - o) / o.replace(0, np.nan)
    d["retorno_target"] = c.shift(-horizonte) / c - 1
    d["retorno_diario"] = c.pct_change()
    return d

def detectores_heuristicos(df):
    c, v, h, l = df["Close"], df["Volume"], df["High"], df["Low"]
    high_50 = c.rolling(50).max()
    is_breakout = bool(c.iloc[-1] >= high_50.iloc[-2]) if len(c) > 50 else False
    is_inst_acc = bool(v.iloc[-1] > v.rolling(20).mean().iloc[-1] * 2) if len(v) > 20 else False
    is_expl_mom = bool(c.pct_change(20).iloc[-1] > 0.15) if len(c) > 20 else False
    
    is_fibo_golden = False
    if len(df) > 60:
        max_60, min_60 = h.iloc[-60:].max(), l.iloc[-60:].min()
        rango = max_60 - min_60
        if rango > 0:
            f50, f618 = max_60 - (rango * 0.500), max_60 - (rango * 0.618)
            if min(f50, f618) * 0.985 <= c.iloc[-1] <= max(f50, f618) * 1.015:
                is_fibo_golden = True

    return is_breakout, is_inst_acc, is_expl_mom, is_fibo_golden

def contexto_vix(vix):
    for n, (lo, hi, f, i, c) in VIX_CONTEXTOS.items():
        if lo <= vix < hi: return n, f, i, c
    return "OPTIMISMO", 1.05, "🔵", "#60a5fa"

# ─────────────────────────────────────────────────────────────────
# MOTOR OLS Y WALK-FORWARD
# ─────────────────────────────────────────────────────────────────
def _normalizar(X_tr: np.ndarray, x_pred: np.ndarray):
    mu, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    std[std == 0] = 1.0
    return (X_tr - mu) / std, (x_pred - mu) / std

def _walk_forward_features(d: pd.DataFrame, feats: list, y_full: np.ndarray, N: int, inicio_wf: int):
    X_full, k = d[feats].values, len(feats)
    preds, pesos = np.full(N, np.nan), np.full(N, 0.0)

    for i in range(inicio_wf, N):
        fin, inicio = i - BLIND_SPOT, max(LAG_INICIAL, i - BLIND_SPOT - VENTANA_TRAIN)
        X_tr, y_tr = X_full[inicio:fin], y_full[inicio:fin]
        mask = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_tr)
        n_valido = mask.sum()
        if n_valido < 50 or not np.all(np.isfinite(X_full[i])): continue

        X_norm, x_norm = _normalizar(X_tr[mask], X_full[i])
        Xt, xc_hoy = np.column_stack([X_norm, np.ones(n_valido)]), np.append(x_norm, 1.0)
        
        coefs, _, _, _ = np.linalg.lstsq(Xt, y_tr[mask], rcond=None)
        yp = Xt @ coefs
        sstot = np.sum((y_tr[mask] - y_tr[mask].mean()) ** 2)
        if sstot <= 0: continue

        r2_c = 1.0 - np.sum((y_tr[mask] - yp) ** 2) / sstot
        r2_adj = 1.0 - (1.0 - r2_c) * (n_valido - 1) / (n_valido - k - 1)
        if r2_c <= 0 or (r2_c / k) / ((1.0 - r2_c) / (n_valido - k - 1)) < F_UMBRAL or r2_adj < R2_MIN: continue

        preds[i], pesos[i] = float(xc_hoy @ coefs), r2_adj

    return float(preds[-1]) if np.isfinite(preds[-1]) else 0.0, preds, float(pesos[-1]), pesos

def ejecutar_modelo(d: pd.DataFrame, horizonte: int):
    vacio = dict(pred_rsi=0, pred_macd=0, pred_medias=0, consenso=0, r2_rsi=0, r2_macd=0, r2_medias=0, r2_prom=0)
    d["consenso_raw"] = np.nan
    N, inicio_wf = len(d), LAG_INICIAL + VENTANA_TRAIN + BLIND_SPOT
    if N < inicio_wf + 10: return vacio, d

    y_full = d["retorno_target"].values
    p1, h1, w1, pw1 = _walk_forward_features(d, FEATS_M1, y_full, N, inicio_wf)
    p2, h2, w2, pw2 = _walk_forward_features(d, FEATS_M2, y_full, N, inicio_wf)
    p3, h3, w3, pw3 = _walk_forward_features(d, FEATS_M3, y_full, N, inicio_wf)

    peso_hoy = w1 + w2 + w3
    c_hoy, r2_p = ((p1*w1 + p2*w2 + p3*w3) / peso_hoy, peso_hoy / sum(1 for w in [w1, w2, w3] if w > 0)) if peso_hoy > 0 else (0.0, 0.0)

    h1c, h2c, h3c = np.nan_to_num(h1), np.nan_to_num(h2), np.nan_to_num(h3)
    p_hist = pw1 + pw2 + pw3
    with np.errstate(divide="ignore", invalid="ignore"):
        d["consenso_raw"] = np.where(p_hist > 0, (h1c*pw1 + h2c*pw2 + h3c*pw3) / p_hist, np.nan)

    return dict(pred_rsi=round(p1, 6), pred_macd=round(p2, 6), pred_medias=round(p3, 6), consenso=round(c_hoy, 6),
                r2_rsi=round(w1, 4), r2_macd=round(w2, 4), r2_medias=round(w3, 4), r2_prom=round(r2_p, 4)), d

def ejecutar_modelo_multitemporal(d: pd.DataFrame, vix_serie: pd.Series, logger: ErrorLogger, ticker: str):
    N, inicio_wf = len(d), LAG_INICIAL + VENTANA_TRAIN + BLIND_SPOT
    if N < inicio_wf + 10: return None
    c = d["Close"]
    Y_all = np.column_stack([(c.shift(-10)/c-1).values, (c.shift(-20)/c-1).values, (c.shift(-30)/c-1).values])

    def wf_multi(feats: list):
        X_full, k = d[feats].values, len(feats)
        preds_m, pesos_m = np.full((N, 3), np.nan), np.zeros((N, 3))
        for i in range(inicio_wf, N):
            fin, ini = i - BLIND_SPOT, max(LAG_INICIAL, i - BLIND_SPOT - VENTANA_TRAIN)
            X_tr, Y_tr = X_full[ini:fin], Y_all[ini:fin]
            mask = np.all(np.isfinite(X_tr), axis=1) & np.all(np.isfinite(Y_tr), axis=1)
            n_v = mask.sum()
            if n_v < 50 or not np.all(np.isfinite(X_full[i])): continue

            X_norm, x_norm = _normalizar(X_tr[mask], X_full[i])
            Xt, xc_hoy = np.column_stack([X_norm, np.ones(n_v)]), np.append(x_norm, 1.0)
            try: Coefs, _, _, _ = np.linalg.lstsq(Xt, Y_tr[mask], rcond=None)
            except np.linalg.LinAlgError: continue

            for j in range(3):
                yp, yt_j = Xt @ Coefs[:, j], Y_tr[mask][:, j]
                sstot = np.sum((yt_j - yt_j.mean()) ** 2)
                if sstot <= 0: continue
                r2_c = 1.0 - np.sum((yt_j - yp) ** 2) / sstot
                r2_adj = 1.0 - (1.0 - r2_c) * (n_v - 1) / (n_v - k - 1)
                if r2_c <= 0 or (r2_c/k)/((1.0-r2_c)/(n_v-k-1)) < F_UMBRAL or r2_adj < R2_MIN: continue
                preds_m[i, j], pesos_m[i, j] = float(xc_hoy @ Coefs[:, j]), r2_adj
        return preds_m, pesos_m

    preds1, pesos1 = wf_multi(FEATS_M1)
    preds2, pesos2 = wf_multi(FEATS_M2)
    preds3, pesos3 = wf_multi(FEATS_M3)

    vix_hoy = float(d["vix"].iloc[-1]) if "vix" in d.columns and pd.notna(d["vix"].iloc[-1]) else 18.0
    _, ctx_fac, _, _ = contexto_vix(vix_hoy)

    fuerzas, r2s = [], []
    for j in range(3):
        w1, w2, w3 = pesos1[-1, j], pesos2[-1, j], pesos3[-1, j]
        p1, p2, p3 = np.nan_to_num(preds1[-1, j]), np.nan_to_num(preds2[-1, j]), np.nan_to_num(preds3[-1, j])
        pt = w1 + w2 + w3
        fuerzas.append(((p1*w1 + p2*w2 + p3*w3) / pt * ctx_fac) if pt > 0 else 0.0)
        r2s.append((pt / sum(1 for w in [w1,w2,w3] if w > 0)) if pt > 0 else 0.0)

    if max(r2s) < R2_MIN: return None
    vc, vv, fm = sum(1 for f in fuerzas if f > 0.02), sum(1 for f in fuerzas if f < -0.02), float(np.mean(fuerzas))
    
    if vc == 3: señal = "COMPRA FUERTE (3/3)"
    elif vc >= 2 and fm > 0.02: señal = "COMPRAR"
    elif vv == 3: señal = "VENTA FUERTE (3/3)"
    elif vv >= 2 and fm < -0.02: señal = "VENDER"
    else: señal = "ESPERAR / MIXTO"

    pw20 = pesos1[:,1] + pesos2[:,1] + pesos3[:,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cons_h20 = np.where(pw20 > 0, (np.nan_to_num(preds1[:,1])*pesos1[:,1] + np.nan_to_num(preds2[:,1])*pesos2[:,1] + np.nan_to_num(preds3[:,1])*pesos3[:,1]) / pw20, np.nan)
    
    conds_vix = [d["vix"]<15, (d["vix"]>=15)&(d["vix"]<24), (d["vix"]>=24)&(d["vix"]<32), d["vix"]>=32] if "vix" in d.columns else [np.zeros(N,bool)]*4
    cons_f20 = cons_h20 * np.select(conds_vix, [1.0, 1.05, 0.9, 0.75], default=1.0)
    sig_raw = np.where(cons_f20 > 0.02, 1, np.where(cons_f20 < -0.02, -1, 0))
    
    strat_r = (pd.Series(sig_raw).replace(0, np.nan).ffill(limit=19).fillna(0).shift(1) * d["retorno_diario"].values).dropna()
    met = {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0}
    if len(strat_r) > 5:
        met["sharpe"] = float(np.sqrt(252)*strat_r.mean()/strat_r.std()) if strat_r.std() != 0 else 0.0
        d_neg = strat_r[strat_r < 0]
        met["sortino"] = float(np.sqrt(252)*strat_r.mean()/d_neg.std()) if len(d_neg) > 2 and d_neg.std() != 0 else 0.0
        eq = (1 + strat_r).cumprod()
        met["max_dd"] = float(((eq - eq.cummax()) / eq.cummax()).min())

    mask_t = (sig_raw != 0) & np.isfinite(Y_all[:,1])
    wr = float((((sig_raw[mask_t] == 1) & (Y_all[:,1][mask_t] > 0)) | ((sig_raw[mask_t] == -1) & (Y_all[:,1][mask_t] < 0))).sum() / mask_t.sum()) if mask_t.sum() > 0 else 0.0

    return {"señal": señal, "f_10d": round(fuerzas[0], 4), "f_20d": round(fuerzas[1], 4), "f_30d": round(fuerzas[2], 4),
            "fuerza_media": round(fm, 4), "r2_medio": round(float(np.mean(r2s)), 4), "win_rate": wr,
            "sharpe_oos": round(met["sharpe"], 2), "sortino_oos": round(met["sortino"], 2), "max_dd_oos": round(met["max_dd"], 4)}

def calcular_auditoria_mtm(d: pd.DataFrame, vix_serie: pd.Series, horizonte: int):
    df_aud = d.copy()
    df_aud["vix"] = vix_serie.reindex(df_aud.index, method="ffill")
    conds = [df_aud["vix"]<15, (df_aud["vix"]>=15)&(df_aud["vix"]<24), (df_aud["vix"]>=24)&(df_aud["vix"]<32), df_aud["vix"]>=32]
    df_aud["consenso_final"] = df_aud["consenso_raw"] * np.select(conds, [1.0, 1.05, 0.90, 0.75], default=1.0)

    df_aud["señal_h"] = "ESPERAR"
    df_aud.loc[df_aud["consenso_final"] >  0.02, "señal_h"] = "COMPRAR"
    df_aud.loc[df_aud["consenso_final"] < -0.02, "señal_h"] = "VENDER"

    trades = df_aud[df_aud["señal_h"] != "ESPERAR"].dropna(subset=["retorno_target"]).copy()
    met = {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "win_rate": 0.0}
    
    if not trades.empty:
        trades["resultado"] = np.where(((trades["señal_h"] == "COMPRAR") & (trades["retorno_target"] > 0)) |
                                       ((trades["señal_h"] == "VENDER")  & (trades["retorno_target"] < 0)), "✅ ACIERTO", "❌ FALLO")
        met["win_rate"] = float((trades["resultado"] == "✅ ACIERTO").sum() / len(trades))
        trades = trades[["Close", "vix", "rsi", "consenso_final", "señal_h", "retorno_target", "resultado"]].sort_index(ascending=False)

    r_diario = (df_aud["señal_h"].map({"COMPRAR":1, "VENDER":-1, "ESPERAR":0}).replace(0, np.nan).ffill(limit=horizonte-1).fillna(0).shift(1) * df_aud["retorno_diario"]).dropna()
    if len(r_diario) > 5:
        met["sharpe"] = float(np.sqrt(252) * r_diario.mean() / r_diario.std()) if r_diario.std() != 0 else 0.0
        d_neg = r_diario[r_diario < 0]
        met["sortino"] = float(np.sqrt(252) * r_diario.mean() / d_neg.std()) if len(d_neg) > 2 and d_neg.std() != 0 else 0.0
        eq = (1 + r_diario).cumprod()
        met["max_dd"] = float(((eq - eq.cummax()) / eq.cummax()).min())
        df_aud["equity_curve"] = eq

    return trades, df_aud, met

# ─────────────────────────────────────────────────────────────────
# UI Y SESSION STATE
# ─────────────────────────────────────────────────────────────────
for key in ["df_rank", "rank_mercado", "rank_anios", "df_errores"]:
    if key not in st.session_state: st.session_state[key] = None

with st.sidebar:
    st.markdown(f"👤 **Usuario Activo:** `{usuario_actual}`")
    if st.button("Cerrar Sesión", use_container_width=True):
        st.session_state["password_correct"] = False
        st.rerun()
    st.markdown("---")
    st.markdown("## ⚙️ Panel de Control")
    mercado   = st.radio("Mercado", ["🇺🇸 Estados Unidos", "🇦🇷 Argentina (Merval)"])
    lista     = cargar_universo_usa() if "Unidos" in mercado else cargar_universo_arg()
    ticker    = st.selectbox("Activo Individual", lista)
    anios     = st.slider("Años de historia", 2, 10, 3)
    horizonte = st.slider("Horizonte de predicción (días)", 5, 60, 20, step=5)
    st.markdown("---")
    show_bb    = st.checkbox("Bandas de Bollinger", True)
    show_vol   = st.checkbox("Volumen", True)
    show_stoch = st.checkbox("Estocástico %K/%D", False)
    show_atr   = st.checkbox("ATR (%)", False)
    st.markdown("---")
    if st.button("🔄 Limpiar caché", use_container_width=True):
        st.cache_data.clear()
        st.session_state["df_rank"], st.session_state["df_errores"] = None, None
        st.rerun()

st.markdown("# 📊 Modelo IA Screener")
st.markdown(f"`{ticker}` · {mercado} · Objetivo: **{horizonte} días**")
st.markdown("---")

with st.spinner(f"Procesando {ticker}..."):
    df_raw, err_m, err_d = descargar(ticker, anios)
    vix_s   = descargar_vix(anios)
    bench_s = descargar_benchmark(mercado, anios)

if df_raw is None: st.error(f"⚠️ {err_m}: {err_d}"); st.stop()

d = calcular_indicadores(df_raw, bench_s, horizonte)
d["vix"] = vix_s.reindex(d.index, method="ffill")
modelo_res, d = ejecutar_modelo(d, horizonte)
bk, inst, expl, fibo = detectores_heuristicos(df_raw)

hoy = d.iloc[-1]
v_hoy = float(hoy.get("vix", 18.0) or 18.0)
ctx_nom, ctx_fac, ctx_ico, _ = contexto_vix(v_hoy)
cons_adj = modelo_res["consenso"] * ctx_fac
señal_hoy = "COMPRAR" if cons_adj > 0.02 else ("VENDER" if cons_adj < -0.02 else "ESPERAR")
sig_color = {"COMPRAR": "#34d399", "VENDER": "#f87171", "ESPERAR": "#facc15"}[señal_hoy]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(f"<div style='background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;text-align:center'><div style='font-size:11px;color:#64748b;text-transform:uppercase'>Señal {horizonte}d</div><div style='font-size:1.7rem;font-weight:700;color:{sig_color}'>{señal_hoy}</div><div style='font-size:12px;color:#64748b'>{'+' if cons_adj >= 0 else ''}{cons_adj*100:.2f}%</div></div>", unsafe_allow_html=True)
c2.metric("Precio", f"${hoy['Close']:.2f}")
c3.metric(f"VIX {ctx_ico}", f"{v_hoy:.2f}", ctx_nom, delta_color="off")
c4.metric("RSI-14", f"{hoy['rsi']:.1f}", "Sobrecompra" if hoy["rsi"] > 70 else ("Sobreventa" if hoy["rsi"] < 30 else "Normal"), delta_color="inverse" if hoy["rsi"] > 70 else "normal")
c5.metric("MACD", f"{hoy['macd']:.4f}", "↑ Alcista" if hoy["macd"] > hoy["macd_sig"] else "↓ Bajista", delta_color="normal" if hoy["macd"] > hoy["macd_sig"] else "inverse")
c6.metric("R² Promedio", f"{modelo_res['r2_prom']*100:.1f}%", "Significativo" if modelo_res["r2_prom"] >= R2_MIN else "Sin señal")

if any([bk, inst, expl, fibo]):
    etiquetas = [t for t, cond in zip(["🚀 Breakout (Máx 50d)", "🏦 Acum. Inst. (Vol. 2x)", "🔥 Momentum (>15%)", "📐 Golden Pocket Fib"], [bk, inst, expl, fibo]) if cond]
    st.markdown(f"**Banderas activas:** `{'` · `'.join(etiquetas)}`")

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Gráfico", "🧠 Modelo LINEST", "🕵️ Auditoría OOS", "📋 Datos", "🏆 Ranking Global", "💼 Mi Cartera"])

with tab1:
    rows_n  = 1 + sum([show_vol, show_stoch, show_atr])
    heights = [0.55] + [0.15] * (rows_n - 1)
    subs    = (["Precio"] + (["Volumen"] if show_vol else []) + (["Estocástico"] if show_stoch else []) + (["ATR %"] if show_atr else []))
    fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=heights, subplot_titles=subs)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Precio", increasing_line_color="#34d399", decreasing_line_color="#f87171", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["mm10"], name="MM10", line=dict(color="#facc15", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["mm50"], name="MM50", line=dict(color="#f87171", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema12"], name="EMA12", line=dict(color="#a78bfa", width=1, dash="dash"), visible="legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema26"], name="EMA26", line=dict(color="#818cf8", width=1, dash="dash"), visible="legendonly"), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], name="BB+", line=dict(color="rgba(148,163,184,0.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], name="BB-", fill="tonexty", fillcolor="rgba(148,163,184,0.07)", line=dict(color="rgba(148,163,184,0.4)", width=1)), row=1, col=1)
    if cons_adj != 0.0 and modelo_res["r2_prom"] >= R2_MIN:
        obj = hoy["Close"] * (1 + cons_adj)
        fig.add_hline(y=obj, line_dash="dash", line_color=sig_color, opacity=0.8, annotation_text=f"Objetivo {horizonte}d: ${obj:.2f} ({'+' if cons_adj>=0 else ''}{cons_adj*100:.1f}%)", annotation_font_color=sig_color, row=1, col=1)
    cur = 2
    if show_vol:
        cv = ["#34d399" if c >= o else "#f87171" for c, o in zip(d["Close"], d["Open"])]
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=cv, showlegend=False), row=cur, col=1)
        cur += 1
    if show_stoch and "stoch_k" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["stoch_k"], name="%K", line=dict(color="#60a5fa", width=1.5)), row=cur, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["stoch_d"], name="%D", line=dict(color="#f472b6", width=1.5, dash="dot")), row=cur, col=1)
        cur += 1
    if show_atr and "atr_pct" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["atr_pct"] * 100, name="ATR %", fill="tozeroy", fillcolor="rgba(167,139,250,0.15)", line=dict(color="#a78bfa", width=1.5)), row=cur, col=1)

    fig.update_layout(height=580 + (rows_n - 1) * 120, xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", margin=dict(t=30, b=10))
    for i in range(1, rows_n + 1):
        fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155", row=i, col=1); fig.update_xaxes(gridcolor="#1e293b", row=i, col=1)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(f"### 🧠 Resultados LINEST Walk-Forward ({horizonte}d)")
    cols = st.columns(3)
    for c, n, p, r, ds in zip(cols, ["🔵 M1: Momentum", "🟣 M2: Divergencia", "🟡 M3: Tendencia"], 
                              [modelo_res["pred_rsi"], modelo_res["pred_macd"], modelo_res["pred_medias"]], 
                              [modelo_res["r2_rsi"], modelo_res["r2_macd"], modelo_res["r2_medias"]], 
                              ["RSI · ATR · F.Rel", "var. MACD · ATR · Gap", "desv. MM50 · Vol"]):
        with c:
            st.markdown(f"#### {n}")
            act = r >= R2_MIN
            st.metric("Predicción", f"{p*100:+.2f}%" if act else "— filtrado")
            st.metric("R² adj", f"{r*100:.2f}%", "✅ Significativo" if act else "❌ Ruido", delta_color="normal" if act else "inverse")
            st.caption(ds)

with tab3:
    st.markdown(f"### 🕵️ Auditoría Out-Of-Sample (Horizonte: {horizonte}d)")
    trades, df_aud, met = calcular_auditoria_mtm(d, vix_s, horizonte)
    if trades.empty: st.info("Sin señales históricas útiles.")
    else:
        ac1, ac2, ac3, ac4, ac5, ac6, ac7 = st.columns(7)
        ac1.metric("Operaciones", len(trades))
        ac2.metric("Aciertos", (trades["resultado"] == "✅ ACIERTO").sum())
        ac3.metric("Fallos", len(trades) - (trades["resultado"] == "✅ ACIERTO").sum())
        ac4.metric("Win Rate", f"{met['win_rate']*100:.1f}%", "Bueno" if met['win_rate']>=0.60 else "Peligroso", delta_color="normal" if met['win_rate']>=0.60 else "inverse")
        ac5.metric("Sharpe MTM", f"{met['sharpe']:.2f}"); ac6.metric("Sortino MTM", f"{met['sortino']:.2f}"); ac7.metric("Max Drawdown", f"{met['max_dd']*100:.1f}%", delta_color="inverse")
        if "equity_curve" in df_aud.columns:
            fig_eq = px.line(df_aud, x=df_aud.index, y="equity_curve", title="Curva de Capital (MTM Diario)")
            fig_eq.add_hline(y=1, line_dash="dash", line_color="#94a3b8")
            fig_eq.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_eq, use_container_width=True)
        st.dataframe(trades.rename(columns={"Close":"Precio","consenso_final":"Consenso", "señal_h":"Señal","retorno_target":f"Ret {horizonte}d","resultado":"Resultado"})
            .style.format({"Precio":"${:.2f}","vix":"{:.1f}","rsi":"{:.1f}", "Consenso":"{:+.3f}",f"Ret {horizonte}d":"{:+.2%}"}), use_container_width=True, height=280)

with tab4:
    st.markdown("### 📋 Últimas 100 filas de datos")
    st.dataframe(d[[c for c in ["Close", "Volume", "rsi", "macd", "mm10", "mm50", "atr_pct", "fuerza_rel", "vix", "consenso_raw"] if c in d.columns]].tail(100).sort_index(ascending=False).style.format("{:.4f}"), use_container_width=True, height=450)

with tab5:
    st.markdown("### 🏆 Ranking Multitemporal Acelerado")
    if st.button(f"🚀 Ejecutar Escaneo Rápido ({mercado})", type="primary"):
        barra, logger, resultados = st.progress(0, "Iniciando escaneo..."), ErrorLogger(), []
        min_vol, min_price = (1_000_000, 5.0) if "Unidos" in mercado else (10_000, 0.0)
        for i, activo in enumerate(lista):
            barra.progress((i+1)/len(lista), f"Evaluando {activo} ({i+1}/{len(lista)})")
            df_act, err_m, err_d = descargar(activo, anios)
            if df_act is None: logger.add(activo, err_m, err_d); continue
            if df_act["Close"].iloc[-1] < min_price or df_act["Volume"].iloc[-20:].mean() < min_vol: continue
            
            d_base = calcular_indicadores(df_act, bench_s, 20)
            d_base["vix"] = vix_s.reindex(d_base.index, method="ffill")
            mod_res = ejecutar_modelo_multitemporal(d_base, vix_s, logger, activo)
            if mod_res is None: continue
            
            bk, inst, expl, fibo = detectores_heuristicos(df_act)
            tags = [t for t, cond in zip(["🚀 Breakout", "🏦 Inst. Acc.", "🔥 Momentum", "📐 Fibo"], [bk, inst, expl, fibo]) if cond]
            mod_res.update({"Activo": activo, "Precio": round(float(df_act["Close"].iloc[-1]), 2), "Banderas": " | ".join(tags) if tags else "—"})
            resultados.append(mod_res)
            
        barra.empty()
        if resultados:
            st.session_state["df_rank"] = pd.DataFrame(resultados)[["Activo", "Precio", "señal", "f_10d", "f_20d", "f_30d", "fuerza_media", "r2_medio", "win_rate", "sharpe_oos", "sortino_oos", "max_dd_oos", "Banderas"]].sort_values("fuerza_media", ascending=False).reset_index(drop=True)
            st.session_state["rank_mercado"], st.session_state["rank_anios"] = mercado, anios
            st.success(f"✅ {len(resultados)} activos validados.")
        st.session_state["df_errores"] = logger.to_df()

    if st.session_state["df_rank"] is not None:
        df_show = st.session_state["df_rank"].rename(columns={"señal":"Señal", "f_10d":"F(10d)", "f_20d":"F(20d)", "f_30d":"F(30d)", "fuerza_media":"Fuerza Media", "r2_medio":"R² Medio", "win_rate":"Win Rate", "sharpe_oos":"Sharpe", "sortino_oos":"Sortino", "max_dd_oos":"Max DD"})
        f1, f2, f3 = st.columns(3)
        filtro_s = f1.multiselect("Filtrar señal", ["COMPRA FUERTE (3/3)", "COMPRAR", "ESPERAR / MIXTO", "VENDER", "VENTA FUERTE (3/3)"], default=["COMPRA FUERTE (3/3)", "COMPRAR", "VENTA FUERTE (3/3)", "VENDER"])
        df_show = df_show[df_show["Señal"].isin(filtro_s) & (df_show["R² Medio"] >= f2.slider("R² prom min (%)", 0, 20, 1)/100) & (df_show["Win Rate"] >= f3.slider("Win Rate min (%)", 0, 100, 60)/100)]
        
        def c_s(v): return "color:#10b981;font-weight:bold" if "COMPRA FUERTE" in v else ("color:#34d399;font-weight:bold" if "COMPRAR" in v else ("color:#ef4444;font-weight:bold" if "VENTA FUERTE" in v else ("color:#f87171;font-weight:bold" if "VENDER" in v else "color:#facc15")))
        st.dataframe(df_show.style.map(c_s, subset=["Señal"]).format({"Precio": "${:.2f}", "F(10d)": "{:+.2%}", "F(20d)": "{:+.2%}", "F(30d)": "{:+.2%}", "Fuerza Media": "{:+.2%}", "R² Medio": "{:.2%}", "Win Rate": "{:.1%}", "Max DD": "{:.1%}"}), use_container_width=True, height=480, hide_index=True)

with tab6:
    st.markdown("### 💼 Gestión de Cartera Multi-Usuario (Google Sheets)")
    st.markdown(f"Bienvenido/a, **{usuario_actual}**. Este es tu portafolio personal. Las operaciones que ingreses aquí no se mezclarán con las de otros usuarios del sistema.")
    from streamlit_gsheets import GSheetsConnection
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    try:
        df_completo = conn.read(worksheet="Sheet1")
        df_completo = df_completo.dropna(how="all")
        
        for col in ["Usuario", "Activo", "Fecha_Compra", "Precio_Compra", "Horizonte_Dias", "Estado", "Fecha_Cierre", "Precio_Cierre", "Resultado_Pct"]:
            if col not in df_completo.columns: df_completo[col] = None
                
        df_completo["Estado"] = df_completo["Estado"].fillna("ABIERTA")
        df_completo["Usuario"] = df_completo["Usuario"].fillna("admin")
        df_completo["Precio_Compra"] = pd.to_numeric(df_completo["Precio_Compra"], errors="coerce")
        df_completo["Resultado_Pct"] = pd.to_numeric(df_completo["Resultado_Pct"], errors="coerce")
        
        # Filtro maestro de multi-usuario
        df_cartera = df_completo[df_completo["Usuario"] == usuario_actual].copy()
        
    except Exception as e:
        st.error(f"Error de lectura. Configure los Secrets de Google. Detalles: {e}")
        df_completo = pd.DataFrame(columns=["Usuario", "Activo", "Fecha_Compra", "Precio_Compra", "Horizonte_Dias", "Estado", "Fecha_Cierre", "Precio_Cierre", "Resultado_Pct"])
        df_cartera = df_completo.copy()

    df_abiertas = df_cartera[df_cartera["Estado"] == "ABIERTA"]
    df_cerradas = df_cartera[df_cartera["Estado"] == "CERRADA"]

    with st.expander("➕ Abrir Nueva Posición", expanded=False):
        with st.form("form_cartera"):
            c_act, c_precio, c_fecha, c_horiz = st.columns(4)
            n_activo = c_act.selectbox("Ticker", sorted(list(set(cargar_universo_usa() + cargar_universo_arg()))))
            n_precio = c_precio.number_input("Precio Compra ($)", min_value=0.01, step=0.5, format="%.2f")
            n_fecha  = c_fecha.date_input("Fecha de Compra")
            n_horiz  = c_horiz.selectbox("Horizonte Objetivo", [10, 20, 30])
            if st.form_submit_button("Impactar en Base de Datos") and n_activo:
                nueva_fila = pd.DataFrame([{"Usuario": usuario_actual, "Activo": n_activo, "Fecha_Compra": n_fecha.strftime("%Y-%m-%d"), "Precio_Compra": float(n_precio), "Horizonte_Dias": int(n_horiz), "Estado": "ABIERTA", "Fecha_Cierre": None, "Precio_Cierre": None, "Resultado_Pct": None}])
                conn.update(worksheet="Sheet1", data=pd.concat([df_completo, nueva_fila], ignore_index=True))
                st.success(f"✅ {n_activo} registrada para {usuario_actual}."); st.cache_data.clear(); st.rerun()

    if not df_abiertas.empty:
        st.markdown("#### 📊 Posiciones Activas")
        if st.button("🔄 Ejecutar Auditoría en Vivo", type="primary"):
            barra_c, res_c = st.progress(0, "Calculando..."), []
            h_f, b_c, v_c, l_c = datetime.today().date(), descargar_benchmark(mercado, anios), descargar_vix(anios), ErrorLogger()

            for idx, row in df_abiertas.iterrows():
                barra_c.progress((list(df_abiertas.index).index(idx) + 1) / len(df_abiertas), f"Analizando {row['Activo']}...")
                try:
                    df_act, _, _ = descargar(row["Activo"], 2)
                    if df_act is None: continue
                    p_act, p_comp = float(df_act["Close"].iloc[-1]), float(row["Precio_Compra"])
                    d_rest = int(row["Horizonte_Dias"]) - np.busday_count(datetime.strptime(str(row["Fecha_Compra"]), "%Y-%m-%d").date(), h_f)
                    
                    d_base = calcular_indicadores(df_act, b_c, 20)
                    d_base["vix"] = v_c.reindex(d_base.index, method="ffill")
                    mod_res = ejecutar_modelo_multitemporal(d_base, v_c, l_c, row["Activo"])
                    res_c.append({"Activo": row["Activo"], "Fecha Compra": row["Fecha_Compra"], "Horiz": f"{row['Horizonte_Dias']}d", "Compra": round(p_comp, 2), "Actual": round(p_act, 2), "P&L": (p_act/p_comp)-1, "Días Restantes": f"⏳ {d_rest}d" if d_rest > 0 else "🚨 CERRAR HOY", "Señal HOY": mod_res["señal"] if mod_res else "RUIDO"})
                except Exception: pass
            
            barra_c.empty()
            if res_c:
                def s_c(r):
                    s = [''] * len(r)
                    ip = r.index.get_loc('P&L'); s[ip] = 'color: #34d399; font-weight: bold' if r['P&L'] > 0 else 'color: #f87171; font-weight: bold'
                    idr = r.index.get_loc('Días Restantes'); s[idr] = 'background-color: #ef4444; color: white; font-weight: bold' if "CERRAR" in r['Días Restantes'] else ''
                    sig = r.index.get_loc('Señal HOY'); s[sig] = 'color: #34d399' if "COMPRA" in str(r['Señal HOY']) else ('color: #f87171' if "VENTA" in str(r['Señal HOY']) else 'color: #facc15')
                    return s
                st.dataframe(pd.DataFrame(res_c).style.apply(s_c, axis=1).format({"Compra": "${:.2f}", "Actual": "${:.2f}", "P&L": "{:+.2%}"}), use_container_width=True, hide_index=True)

        st.markdown("#### ❌ Cerrar Posición")
        with st.form("form_cierre"):
            c1, c2, _ = st.columns(3)
            ticker_cierre = c1.selectbox("Activo a liquidar", df_abiertas["Activo"].unique().tolist())
            precio_cierre = c2.number_input("Precio de Venta ($)", min_value=0.01, step=0.5, format="%.2f")
            
            if st.form_submit_button("Liquidar Activo") and ticker_cierre:
                idx_real = df_abiertas[df_abiertas["Activo"] == ticker_cierre].index[0]
                p_comp_orig = float(df_completo.at[idx_real, "Precio_Compra"])
                df_completo.at[idx_real, "Estado"], df_completo.at[idx_real, "Fecha_Cierre"], df_completo.at[idx_real, "Precio_Cierre"], df_completo.at[idx_real, "Resultado_Pct"] = "CERRADA", datetime.today().strftime("%Y-%m-%d"), precio_cierre, (precio_cierre / p_comp_orig) - 1
                conn.update(worksheet="Sheet1", data=df_completo)
                st.success(f"✅ {ticker_cierre} liquidado y enviado al historial."); st.cache_data.clear(); st.rerun()

    st.markdown("---")
    if not df_cerradas.empty:
        st.markdown("#### 📜 Historial de Operaciones")
        m1, m2, m3 = st.columns(3)
        m1.metric("Operaciones", len(df_cerradas))
        m2.metric("Win Rate", f"{(df_cerradas['Resultado_Pct'] > 0).sum() / len(df_cerradas) * 100:.1f}%")
        m3.metric("P&L Acumulado", f"{df_cerradas['Resultado_Pct'].sum() * 100:+.2f}%", delta_color="normal" if df_cerradas['Resultado_Pct'].sum() > 0 else "inverse")
        
        st.dataframe(df_cerradas[["Activo", "Fecha_Compra", "Precio_Compra", "Fecha_Cierre", "Precio_Cierre", "Resultado_Pct"]].style.map(lambda v: 'color: #34d399; font-weight: bold' if isinstance(v, float) and v > 0 and v < 1 else ('color: #f87171; font-weight: bold' if isinstance(v, float) and v < 0 else ''), subset=['Resultado_Pct']).format({"Precio_Compra": "${:.2f}", "Precio_Cierre": "${:.2f}", "Resultado_Pct": "{:+.2%}"}), use_container_width=True, hide_index=True)

    with st.expander("⚠️ Zona de Peligro (Precaución)", expanded=False):
        st.error("Eliminará TODAS TUS operaciones de la base de datos. No afectará a otros usuarios.")
        if st.button("🗑️ Vaciar Mi Cartera"):
            df_restante = df_completo[df_completo["Usuario"] != usuario_actual]
            conn.update(worksheet="Sheet1", data=df_restante)
            st.cache_data.clear(); st.rerun()

st.markdown("---")
st.caption("Motor: LINEST Walk-Forward Ortogonal · OLS Multitemporal · Golden Pocket · Multi-Usuario")

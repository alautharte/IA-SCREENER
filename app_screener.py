"""
app.py — Modelo IA Screener (USA & ARG)
Motor LINEST Walk-Forward Ortogonal · OLS Multitemporal · Golden Pocket Fibonacci · Conexión Google Sheets
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
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Modelo IA Screener", page_icon="📊", layout="wide")

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

# Features Completas
FEATS_M1 = ["rsi",      "atr_pct", "fuerza_rel", "ret_1d", "ret_3d"]
FEATS_M2 = ["macd_var", "atr_pct", "fuerza_rel", "ret_5d", "gap_oc"]
FEATS_M3 = ["mm50_var", "mm10_vs_mm50", "vol_var20", "ret_3d", "gap_oc"]

# ─────────────────────────────────────────────────────────────────
# UNIVERSOS DINÁMICOS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def cargar_universo_usa():
    sp500, ndx = [], []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    try:
        sp_table = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            storage_options=headers
        )[0]
        sp500 = [t.replace('.', '-') for t in sp_table['Symbol'].tolist()]
    except Exception:
        pass 

    try:
        ndx_table = pd.read_html(
            'https://en.wikipedia.org/wiki/Nasdaq-100',
            storage_options=headers
        )[4]
        ndx = [t.replace('.', '-') for t in ndx_table['Ticker'].tolist()]
    except (IndexError, KeyError, ValueError):
        pass

    universo = sorted(list(set(sp500 + ndx)))
    if not universo:
        return sorted([
            "AAPL","NVDA","TSLA","META","AMZN","MSFT","GOOGL","PLTR","AMD","NFLX",
            "JPM","BAC","V","MA","BLK","PYPL","SPGI","JNJ","UNH","PG","KO","PEP",
            "WMT","COST","MCD","ABBV","LLY","MRK","XOM","CVX","BA","CAT","HON","GE",
            "INTC","MU","AVGO","TXN","QCOM","AMAT","CSCO","SMCI","MARA","RKLB"
        ])
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

# ─────────────────────────────────────────────────────────────────
# LOGGER DE ERRORES
# ─────────────────────────────────────────────────────────────────
class ErrorLogger:
    def __init__(self):
        self.logs = []
    def add(self, ticker, motivo, detalle=""):
        self.logs.append({"Ticker": ticker, "Motivo": motivo, "Detalle": detalle})
    def to_df(self):
        return pd.DataFrame(self.logs) if self.logs else pd.DataFrame(columns=["Ticker", "Motivo", "Detalle"])
    def __len__(self):
        return len(self.logs)

# ─────────────────────────────────────────────────────────────────
# DESCARGAS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def descargar(ticker, years):
    try:
        df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
        if df is None or df.empty:
            return None, "Sin datos", "Respuesta vacía"
        if len(df) < 100:
            return None, "Datos insuficientes", f"n={len(df)} < 100"
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].copy(), None, None
    except Exception as e:
        return None, "Error API", str(e)[:60]

@st.cache_data(ttl=900, show_spinner=False)
def descargar_vix(years):
    try:
        df = yf.Ticker("^VIX").history(period=f"{years}y")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df["Close"].ffill()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=900, show_spinner=False)
def descargar_benchmark(mercado, years):
    bench = "SPY" if "Unidos" in mercado else "GGAL.BA"
    try:
        df = yf.Ticker(bench).history(period=f"{years}y", auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df["Close"].ffill()
    except Exception:
        return pd.Series(dtype=float)

# ─────────────────────────────────────────────────────────────────
# INDICADORES & DETECTORES
# ─────────────────────────────────────────────────────────────────
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def rsi_wilder(precio, periodo=14):
    d   = precio.diff()
    gan = d.clip(lower=0).ewm(alpha=1/periodo, adjust=False).mean()
    per = (-d).clip(lower=0).ewm(alpha=1/periodo, adjust=False).mean()
    rs  = gan / per.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger(precio, w=20, std=2.0):
    m = precio.rolling(w).mean()
    s = precio.rolling(w).std()
    return m + std*s, m, m - std*s

def atr_calc(high, low, close, w=14):
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/w, adjust=False).mean()

def stochastic(high, low, close, k=14, d=3):
    lo    = low.rolling(k).min()
    hi    = high.rolling(k).max()
    pct_k = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    return pct_k, pct_k.rolling(d).mean()

def calcular_indicadores(df, bench_serie, horizonte=20):
    d = df.copy()
    c, v, o = d["Close"], d["Volume"], d["Open"]

    if not bench_serie.empty:
        b = bench_serie.reindex(d.index, method="ffill")
        d["fuerza_rel"] = c.pct_change(20) - b.pct_change(20)
    else:
        d["fuerza_rel"] = 0.0

    d["ema12"]        = ema(c, 12)
    d["ema26"]        = ema(c, 26)
    d["macd"]         = d["ema12"] - d["ema26"]
    d["macd_sig"]     = ema(d["macd"], 9)
    d["macd_var"]     = (d["macd"] - d["macd_sig"]) / c.replace(0, np.nan)
    d["rsi"]          = rsi_wilder(c)
    d["mm10"]         = c.rolling(10).mean()
    d["mm50"]         = c.rolling(50).mean()
    d["mm50_var"]     = (c - d["mm50"]) / d["mm50"].replace(0, np.nan)
    d["mm10_vs_mm50"] = (d["mm10"] - d["mm50"]) / d["mm50"].replace(0, np.nan)
    d["vol_var20"]    = (v - v.rolling(20).mean()) / v.rolling(20).mean()
    d["bb_upper"], d["bb_mid"], d["bb_lower"] = bollinger(c)
    d["bb_pct"]       = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)
    d["volatilidad"]  = c.pct_change().rolling(20).std() * np.sqrt(252)

    if all(col in d.columns for col in ["High", "Low"]):
        d["atr"]     = atr_calc(d["High"], d["Low"], c)
        d["atr_pct"] = d["atr"] / c.replace(0, np.nan)
        d["stoch_k"], d["stoch_d"] = stochastic(d["High"], d["Low"], c)
    else:
        d["atr_pct"] = np.nan

    d["ret_1d"] = c.pct_change(1)
    d["ret_3d"] = c.pct_change(3)
    d["ret_5d"] = c.pct_change(5)
    d["gap_oc"] = (c - o) / o.replace(0, np.nan)

    d["retorno_target"] = c.shift(-horizonte) / c - 1
    d["retorno_diario"] = c.pct_change()
    return d

def detectores_heuristicos(df):
    c, v, h, l = df["Close"], df["Volume"], df["High"], df["Low"]
    
    # Heurísticas estándar
    high_50     = c.rolling(50).max()
    is_breakout = bool(c.iloc[-1] >= high_50.iloc[-2]) if len(c) > 50 else False
    vol_m20     = v.rolling(20).mean()
    is_inst_acc = bool(v.iloc[-1] > vol_m20.iloc[-1] * 2) if len(v) > 20 else False
    is_expl_mom = bool(c.pct_change(20).iloc[-1] > 0.15) if len(c) > 20 else False
    
    # Análisis FIBONACCI (Golden Pocket 50% - 61.8%)
    is_fibo_golden = False
    if len(df) > 60:
        max_60 = h.iloc[-60:].max()
        min_60 = l.iloc[-60:].min()
        rango  = max_60 - min_60
        
        # Si hay rango operable
        if rango > 0:
            fib_50 = max_60 - (rango * 0.500)
            fib_618 = max_60 - (rango * 0.618)
            
            # Definimos la zona de soporte (Pocket) con 1.5% de tolerancia arriba y abajo
            limite_sup = max(fib_50, fib_618) * 1.015
            limite_inf = min(fib_50, fib_618) * 0.985
            
            precio_actual = c.iloc[-1]
            if limite_inf <= precio_actual <= limite_sup:
                is_fibo_golden = True

    return is_breakout, is_inst_acc, is_expl_mom, is_fibo_golden

def contexto_vix(vix):
    for nombre, (lo, hi, factor, icono, color) in VIX_CONTEXTOS.items():
        if lo <= vix < hi:
            return nombre, factor, icono, color
    return "OPTIMISMO", 1.05, "🔵", "#60a5fa"

# ─────────────────────────────────────────────────────────────────
# NORMALIZACIÓN Z-SCORE
# ─────────────────────────────────────────────────────────────────
def _normalizar(X_tr: np.ndarray, x_pred: np.ndarray):
    mu  = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    return (X_tr - mu) / std, (x_pred - mu) / std

# ─────────────────────────────────────────────────────────────────
# MOTOR WALK-FORWARD INDIVIDUAL (PARA PESTAÑA GRÁFICOS)
# ─────────────────────────────────────────────────────────────────
def _walk_forward_features(d: pd.DataFrame, feats: list, y_full: np.ndarray, N: int, inicio_wf: int):
    X_full = d[feats].values
    k      = len(feats)
    preds  = np.full(N, np.nan)
    pesos  = np.full(N, 0.0)

    for i in range(inicio_wf, N):
        fin    = i - BLIND_SPOT
        inicio = max(LAG_INICIAL, fin - VENTANA_TRAIN)

        X_tr = X_full[inicio:fin]
        y_tr = y_full[inicio:fin]

        mask     = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_tr)
        n_valido = mask.sum()
        if n_valido < 50:
            continue

        X_tr_m = X_tr[mask]
        yt     = y_tr[mask]

        x_hoy = X_full[i]
        if not np.all(np.isfinite(x_hoy)):
            continue

        X_norm, x_norm = _normalizar(X_tr_m, x_hoy)
        Xt     = np.column_stack([X_norm, np.ones(n_valido)])
        xc_hoy = np.append(x_norm, 1.0)

        coefs, _, _, _ = np.linalg.lstsq(Xt, yt, rcond=None)
        yp    = Xt @ coefs
        sstot = np.sum((yt - yt.mean()) ** 2)
        if sstot <= 0: continue

        ssres    = np.sum((yt - yp) ** 2)
        r2_crudo = 1.0 - ssres / sstot
        r2_adj   = 1.0 - (1.0 - r2_crudo) * (n_valido - 1) / (n_valido - k - 1)

        if r2_crudo <= 0: continue

        f_stat = (r2_crudo / k) / ((1.0 - r2_crudo) / (n_valido - k - 1))
        if f_stat < F_UMBRAL or r2_adj < R2_MIN:
            continue

        preds[i] = float(xc_hoy @ coefs)
        pesos[i] = r2_adj

    p_hoy = float(preds[-1]) if np.isfinite(preds[-1]) else 0.0
    w_hoy = float(pesos[-1])
    return p_hoy, preds, w_hoy, pesos

def ejecutar_modelo(d: pd.DataFrame, horizonte: int):
    vacio = dict(pred_rsi=0, pred_macd=0, pred_medias=0,
                 consenso=0, r2_rsi=0, r2_macd=0, r2_medias=0, r2_prom=0)
    d["consenso_raw"] = np.nan

    N = len(d)
    inicio_wf = LAG_INICIAL + VENTANA_TRAIN + BLIND_SPOT
    if N < inicio_wf + 10:
        return vacio, d

    y_full = d["retorno_target"].values

    p1, h1, w1, pw1 = _walk_forward_features(d, FEATS_M1, y_full, N, inicio_wf)
    p2, h2, w2, pw2 = _walk_forward_features(d, FEATS_M2, y_full, N, inicio_wf)
    p3, h3, w3, pw3 = _walk_forward_features(d, FEATS_M3, y_full, N, inicio_wf)

    peso_hoy = w1 + w2 + w3
    if peso_hoy > 0:
        consenso_hoy = (p1*w1 + p2*w2 + p3*w3) / peso_hoy
        r2_prom      = peso_hoy / sum(1 for w in [w1, w2, w3] if w > 0)
    else:
        consenso_hoy, r2_prom = 0.0, 0.0

    peso_hist = pw1 + pw2 + pw3
    h1c = np.where(np.isfinite(h1), h1, 0.0)
    h2c = np.where(np.isfinite(h2), h2, 0.0)
    h3c = np.where(np.isfinite(h3), h3, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        d["consenso_raw"] = np.where(peso_hist > 0, (h1c*pw1 + h2c*pw2 + h3c*pw3) / peso_hist, np.nan)

    return dict(
        pred_rsi=round(p1, 6), pred_macd=round(p2, 6), pred_medias=round(p3, 6),
        consenso=round(consenso_hoy, 6),
        r2_rsi=round(w1, 4), r2_macd=round(w2, 4), r2_medias=round(w3, 4),
        r2_prom=round(r2_prom, 4),
    ), d

# ─────────────────────────────────────────────────────────────────
# MOTOR MULTITEMPORAL MATRICIAL (PARA EL RANKING)
# ─────────────────────────────────────────────────────────────────
def ejecutar_modelo_multitemporal(d: pd.DataFrame, vix_serie: pd.Series, logger: ErrorLogger, ticker: str):
    N = len(d)
    inicio_wf = LAG_INICIAL + VENTANA_TRAIN + BLIND_SPOT
    if N < inicio_wf + 10:
        logger.add(ticker, "Datos insuficientes", f"N={N} < {inicio_wf+10}")
        return None

    c   = d["Close"]
    y10 = (c.shift(-10) / c - 1).values
    y20 = (c.shift(-20) / c - 1).values
    y30 = (c.shift(-30) / c - 1).values
    Y_all = np.column_stack([y10, y20, y30])

    def wf_multi(feats: list):
        X_full = d[feats].values
        k      = len(feats)
        preds_m = np.full((N, 3), np.nan)
        pesos_m = np.zeros((N, 3))

        for i in range(inicio_wf, N):
            fin = i - BLIND_SPOT
            ini = max(LAG_INICIAL, fin - VENTANA_TRAIN)

            X_tr = X_full[ini:fin]
            Y_tr = Y_all[ini:fin]

            mask     = np.all(np.isfinite(X_tr), axis=1) & np.all(np.isfinite(Y_tr), axis=1)
            n_valido = mask.sum()
            if n_valido < 50: continue

            X_tr_m = X_tr[mask]
            Yt     = Y_tr[mask]

            x_hoy = X_full[i]
            if not np.all(np.isfinite(x_hoy)): continue

            X_norm, x_norm = _normalizar(X_tr_m, x_hoy)
            Xt     = np.column_stack([X_norm, np.ones(n_valido)])
            xc_hoy = np.append(x_norm, 1.0)

            try:
                Coefs, _, _, _ = np.linalg.lstsq(Xt, Yt, rcond=None)
            except np.linalg.LinAlgError:
                continue

            for j in range(3):
                yp    = Xt @ Coefs[:, j]
                yt_j  = Yt[:, j]
                sstot = np.sum((yt_j - yt_j.mean()) ** 2)
                if sstot <= 0: continue
                ssres    = np.sum((yt_j - yp) ** 2)
                r2_crudo = 1.0 - ssres / sstot
                r2_adj   = 1.0 - (1.0 - r2_crudo) * (n_valido - 1) / (n_valido - k - 1)
                
                if r2_crudo <= 0: continue
                f_stat = (r2_crudo / k) / ((1.0 - r2_crudo) / (n_valido - k - 1))
                if f_stat < F_UMBRAL or r2_adj < R2_MIN: continue
                
                preds_m[i, j] = float(xc_hoy @ Coefs[:, j])
                pesos_m[i, j] = r2_adj

        return preds_m, pesos_m

    preds1, pesos1 = wf_multi(FEATS_M1)
    preds2, pesos2 = wf_multi(FEATS_M2)
    preds3, pesos3 = wf_multi(FEATS_M3)

    vix_hoy = float(d["vix"].iloc[-1]) if "vix" in d.columns and pd.notna(d["vix"].iloc[-1]) else 18.0
    _, ctx_fac, _, _ = contexto_vix(vix_hoy)

    fuerzas, r2s = [], []
    for j in range(3):
        w1, w2, w3 = pesos1[-1, j], pesos2[-1, j], pesos3[-1, j]
        p1 = preds1[-1, j] if np.isfinite(preds1[-1, j]) else 0.0
        p2 = preds2[-1, j] if np.isfinite(preds2[-1, j]) else 0.0
        p3 = preds3[-1, j] if np.isfinite(preds3[-1, j]) else 0.0
        
        peso_t = w1 + w2 + w3
        consenso = (p1*w1 + p2*w2 + p3*w3) / peso_t if peso_t > 0 else 0.0
        r2_p     = peso_t / sum(1 for w in [w1,w2,w3] if w > 0) if peso_t > 0 else 0.0
        
        fuerzas.append(consenso * ctx_fac)
        r2s.append(r2_p)

    max_r2 = max(r2s)
    if max_r2 < R2_MIN:
        logger.add(ticker, "Rechazo estadístico", f"Max R²={max_r2:.4f} < {R2_MIN}")
        return None

    votos_c = sum(1 for f in fuerzas if f > 0.02)
    votos_v = sum(1 for f in fuerzas if f < -0.02)
    fm      = float(np.mean(fuerzas))

    if votos_c == 3:                 señal = "COMPRA FUERTE (3/3)"
    elif votos_c >= 2 and fm > 0.02: señal = "COMPRAR"
    elif votos_v == 3:               señal = "VENTA FUERTE (3/3)"
    elif votos_v >= 2 and fm < -0.02: señal = "VENDER"
    else:                            señal = "ESPERAR / MIXTO"

    pw20 = pesos1[:,1] + pesos2[:,1] + pesos3[:,1]
    h1_20 = np.where(np.isfinite(preds1[:,1]), preds1[:,1], 0.0)
    h2_20 = np.where(np.isfinite(preds2[:,1]), preds2[:,1], 0.0)
    h3_20 = np.where(np.isfinite(preds3[:,1]), preds3[:,1], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cons_h20 = np.where(pw20 > 0, (h1_20*pesos1[:,1] + h2_20*pesos2[:,1] + h3_20*pesos3[:,1]) / pw20, np.nan)

    conds_vix = [d["vix"]<15, (d["vix"]>=15)&(d["vix"]<24), (d["vix"]>=24)&(d["vix"]<32), d["vix"]>=32] if "vix" in d.columns else [np.zeros(N,bool)]*4
    cons_f20  = cons_h20 * np.select(conds_vix, [1.0, 1.05, 0.9, 0.75], default=1.0)
    
    sig_raw = np.where(cons_f20 > 0.02, 1, np.where(cons_f20 < -0.02, -1, 0))
    pos_activa = pd.Series(sig_raw).replace(0, np.nan).ffill(limit=19).fillna(0)
    strat_diario = pos_activa.shift(1) * d["retorno_diario"].values
    
    strat_r = strat_diario.dropna()
    metricas = {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0}
    if len(strat_r) > 5:
        metricas["sharpe"] = float(np.sqrt(252)*strat_r.mean()/strat_r.std()) if strat_r.std() != 0 else 0.0
        d_neg = strat_r[strat_r < 0]
        metricas["sortino"] = float(np.sqrt(252)*strat_r.mean()/d_neg.std()) if len(d_neg) > 2 and d_neg.std() != 0 else 0.0
        eq = (1 + strat_r).cumprod()
        metricas["max_dd"] = float(((eq - eq.cummax()) / eq.cummax()).min())

    mask_trades = (sig_raw != 0) & np.isfinite(y20)
    if mask_trades.sum() > 0:
        aciertos = ((sig_raw[mask_trades] == 1) & (y20[mask_trades] > 0)) | \
                   ((sig_raw[mask_trades] == -1) & (y20[mask_trades] < 0))
        win_rate = float(aciertos.sum() / mask_trades.sum())
    else:
        win_rate = 0.0

    return {
        "señal":         señal,
        "f_10d":         round(fuerzas[0], 4),
        "f_20d":         round(fuerzas[1], 4),
        "f_30d":         round(fuerzas[2], 4),
        "fuerza_media":  round(fm, 4),
        "r2_medio":      round(float(np.mean(r2s)), 4),
        "win_rate":      win_rate,
        "sharpe_oos":    round(metricas["sharpe"], 2),
        "sortino_oos":   round(metricas["sortino"], 2),
        "max_dd_oos":    round(metricas["max_dd"], 4),
    }

# ─────────────────────────────────────────────────────────────────
# AUDITORÍA MARK-TO-MARKET MÚLTIPLE (INDIVIDUAL)
# ─────────────────────────────────────────────────────────────────
def calcular_auditoria_mtm(d: pd.DataFrame, vix_serie: pd.Series, horizonte: int):
    df_aud = d.copy()
    df_aud["vix"] = vix_serie.reindex(df_aud.index, method="ffill")
    conds = [df_aud["vix"]<15, (df_aud["vix"]>=15)&(df_aud["vix"]<24), (df_aud["vix"]>=24)&(df_aud["vix"]<32), df_aud["vix"]>=32]
    df_aud["consenso_final"] = df_aud["consenso_raw"] * np.select(conds, [1.0, 1.05, 0.90, 0.75], default=1.0)

    df_aud["señal_h"] = "ESPERAR"
    df_aud.loc[df_aud["consenso_final"] >  0.02, "señal_h"] = "COMPRAR"
    df_aud.loc[df_aud["consenso_final"] < -0.02, "señal_h"] = "VENDER"

    trades = df_aud[df_aud["señal_h"] != "ESPERAR"].dropna(subset=["retorno_target"]).copy()
    
    metricas = {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "win_rate": 0.0}
    
    if not trades.empty:
        trades["resultado"] = np.where(
            ((trades["señal_h"] == "COMPRAR") & (trades["retorno_target"] > 0)) |
            ((trades["señal_h"] == "VENDER")  & (trades["retorno_target"] < 0)), "✅ ACIERTO", "❌ FALLO"
        )
        aciertos = (trades["resultado"] == "✅ ACIERTO").sum()
        metricas["win_rate"] = float(aciertos / len(trades))
        trades = trades[["Close", "vix", "rsi", "consenso_final", "señal_h", "retorno_target", "resultado"]].sort_index(ascending=False)

    df_aud["posicion_raw"] = df_aud["señal_h"].map({"COMPRAR": 1, "VENDER": -1, "ESPERAR": 0})
    df_aud["pos_activa"]   = df_aud["posicion_raw"].replace(0, np.nan).ffill(limit=horizonte-1).fillna(0)
    df_aud["strat_diario"] = df_aud["pos_activa"].shift(1) * df_aud["retorno_diario"]
    
    r_diario = df_aud["strat_diario"].dropna()
    
    if len(r_diario) > 5:
        metricas["sharpe"] = float(np.sqrt(252) * r_diario.mean() / r_diario.std()) if r_diario.std() != 0 else 0.0
        d_neg = r_diario[r_diario < 0]
        metricas["sortino"] = float(np.sqrt(252) * r_diario.mean() / d_neg.std()) if len(d_neg) > 2 and d_neg.std() != 0 else 0.0
        eq = (1 + r_diario).cumprod()
        metricas["max_dd"] = float(((eq - eq.cummax()) / eq.cummax()).min())
        df_aud["equity_curve"] = eq

    return trades, df_aud, metricas

# ─────────────────────────────────────────────────────────────────
# SESSION STATE Y UI
# ─────────────────────────────────────────────────────────────────
for key in ["df_rank", "rank_mercado", "rank_anios", "df_errores"]:
    if key not in st.session_state: st.session_state[key] = None

with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")
    st.markdown("---")
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

if df_raw is None:
    st.error(f"⚠️ {err_m}: {err_d}")
    st.stop()

d              = calcular_indicadores(df_raw, bench_s, horizonte)
d["vix"]       = vix_s.reindex(d.index, method="ffill")
modelo_res, d  = ejecutar_modelo(d, horizonte)
bk, inst, expl, fibo = detectores_heuristicos(df_raw)

hoy = d.iloc[-1]
v_hoy = float(hoy.get("vix", 18.0) or 18.0)
ctx_nom, ctx_fac, ctx_ico, _ = contexto_vix(v_hoy)
cons_adj  = modelo_res["consenso"] * ctx_fac
señal_hoy = "COMPRAR" if cons_adj > 0.02 else ("VENDER" if cons_adj < -0.02 else "ESPERAR")
sig_color = {"COMPRAR": "#34d399", "VENDER": "#f87171", "ESPERAR": "#facc15"}[señal_hoy]

# ─────────────────────────────────────────────────────────────────
# MÉTRICAS INDIVIDUALES
# ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(f"<div style='background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;text-align:center'><div style='font-size:11px;color:#64748b;text-transform:uppercase'>Señal {horizonte}d</div><div style='font-size:1.7rem;font-weight:700;color:{sig_color}'>{señal_hoy}</div><div style='font-size:12px;color:#64748b'>{'+' if cons_adj >= 0 else ''}{cons_adj*100:.2f}%</div></div>", unsafe_allow_html=True)
c2.metric("Precio", f"${hoy['Close']:.2f}")
c3.metric(f"VIX {ctx_ico}", f"{v_hoy:.2f}", ctx_nom, delta_color="off")
c4.metric("RSI-14", f"{hoy['rsi']:.1f}", "Sobrecompra" if hoy["rsi"] > 70 else ("Sobreventa" if hoy["rsi"] < 30 else "Normal"), delta_color="inverse" if hoy["rsi"] > 70 else "normal")
c5.metric("MACD", f"{hoy['macd']:.4f}", "↑ Alcista" if hoy["macd"] > hoy["macd_sig"] else "↓ Bajista", delta_color="normal" if hoy["macd"] > hoy["macd_sig"] else "inverse")
c6.metric("R² Promedio", f"{modelo_res['r2_prom']*100:.1f}%", "Significativo" if modelo_res["r2_prom"] >= R2_MIN else "Sin señal")

if any([bk, inst, expl, fibo]):
    etiquetas = []
    if bk:   etiquetas.append("🚀 Breakout (Máx 50d)")
    if inst: etiquetas.append("🏦 Acumulación Institucional (Vol. 2x)")
    if expl: etiquetas.append("🔥 Momentum Explosivo (>15%)")
    if fibo: etiquetas.append("📐 Golden Pocket Fib (50%-61.8%)")
    st.markdown(f"**Banderas activas:** `{'` · `'.join(etiquetas)}`")

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Gráfico", "🧠 Modelo LINEST", "🕵️ Auditoría OOS", "📋 Datos", "🏆 Ranking Global", "💼 Mi Cartera"])

# ══════════════════════ TAB 1: GRÁFICO ══════════════════════
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
        fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155", row=i, col=1)
        fig.update_xaxes(gridcolor="#1e293b", row=i, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════ TAB 2: MODELO ══════════════════════
with tab2:
    st.markdown(f"### 🧠 Resultados LINEST Walk-Forward ({horizonte}d)")
    m1, m2, m3 = st.columns(3)
    for col_ui, nombre, pred, r2, desc in [
        (m1, "🔵 M1: Momentum",    modelo_res["pred_rsi"],    modelo_res["r2_rsi"],    "RSI · ATR · F.Rel · Ret1d · Ret3d"),
        (m2, "🟣 M2: Divergencia", modelo_res["pred_macd"],   modelo_res["r2_macd"],   "var. MACD · ATR · F.Rel · Ret5d · Gap OC"),
        (m3, "🟡 M3: Tendencia",   modelo_res["pred_medias"], modelo_res["r2_medias"], "desv. MM50 · MM10 vs MM50 · var. Vol · Gap OC"),
    ]:
        with col_ui:
            st.markdown(f"#### {nombre}")
            activo = r2 >= R2_MIN
            st.metric("Predicción",    f"{pred*100:+.2f}%" if activo else "— filtrado")
            st.metric("R² adj (peso)", f"{r2*100:.2f}%", "✅ Significativo" if activo else "❌ Ruido — descartado", delta_color="normal" if activo else "inverse")
            st.caption(desc)

# ══════════════════════ TAB 3: AUDITORÍA OOS ══════════════════════
with tab3:
    st.markdown(f"### 🕵️ Auditoría Out-Of-Sample (Horizonte: {horizonte}d)")
    trades, df_aud, metricas = calcular_auditoria_mtm(d, vix_s, horizonte)

    if trades.empty:
        st.info("Sin señales históricas útiles.")
    else:
        total = len(trades)
        aciertos = (trades["resultado"] == "✅ ACIERTO").sum()
        acc = metricas["win_rate"]

        ac1, ac2, ac3, ac4, ac5, ac6, ac7 = st.columns(7)
        ac1.metric("Operaciones", total)
        ac2.metric("Aciertos", aciertos)
        ac3.metric("Fallos", total - aciertos)
        ac4.metric("Win Rate", f"{acc*100:.1f}%", "Bueno" if acc>=0.60 else "Peligroso", delta_color="normal" if acc>=0.60 else "inverse")
        ac5.metric("Sharpe MTM", f"{metricas['sharpe']:.2f}")
        ac6.metric("Sortino MTM", f"{metricas['sortino']:.2f}")
        ac7.metric("Max Drawdown", f"{metricas['max_dd']*100:.1f}%", delta_color="inverse")

        if "equity_curve" in df_aud.columns:
            fig_eq = px.line(df_aud, x=df_aud.index, y="equity_curve", title=f"Curva de Capital (Mark-to-Market Diario)")
            fig_eq.add_hline(y=1, line_dash="dash", line_color="#94a3b8")
            fig_eq.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_eq, use_container_width=True)

        st.dataframe(
            trades.rename(columns={"Close":"Precio","consenso_final":"Consenso", "señal_h":"Señal","retorno_target":f"Retorno {horizonte}d","resultado":"Resultado"})
            .style.format({"Precio":"${:.2f}","vix":"{:.1f}","rsi":"{:.1f}", "Consenso":"{:+.3f}",f"Retorno {horizonte}d":"{:+.2%}"}),
            use_container_width=True, height=280
        )

# ══════════════════════ TAB 4: DATOS ══════════════════════
with tab4:
    st.markdown("### 📋 Últimas 100 filas de datos")
    cols_show = ["Close", "Volume", "rsi", "macd", "mm10", "mm50", "atr_pct", "fuerza_rel", "vix", "consenso_raw"]
    st.dataframe(d[[c for c in cols_show if c in d.columns]].tail(100).sort_index(ascending=False).style.format("{:.4f}"), use_container_width=True, height=450)

# ══════════════════════ TAB 5: RANKING GLOBAL ══════════════════════
with tab5:
    st.markdown("### 🏆 Ranking Multitemporal Acelerado")
    st.markdown(f"**Universo original:** {len(lista)} activos ({mercado}). Se aplicará un filtro de liquidez y precio antes de ejecutar el cálculo matricial pesado.")
    
    if st.session_state["rank_mercado"] != mercado or st.session_state["rank_anios"] != anios:
        st.session_state["df_rank"], st.session_state["df_errores"] = None, None

    if st.button(f"🚀 Ejecutar Escaneo Rápido ({mercado})", type="primary"):
        barra = st.progress(0, text="Iniciando escaneo masivo...")
        logger = ErrorLogger()
        resultados = []
        
        min_vol = 1_000_000 if "Unidos" in mercado else 10_000
        min_price = 5.0 if "Unidos" in mercado else 0.0

        for i, activo in enumerate(lista):
            barra.progress((i+1)/len(lista), text=f"Evaluando {activo} ({i+1}/{len(lista)})")
            
            df_act, err_m, err_d = descargar(activo, anios)
            if df_act is None:
                logger.add(activo, err_m, err_d)
                continue
                
            ult_c = df_act["Close"].iloc[-1]
            ult_v = df_act["Volume"].iloc[-20:].mean()
            
            if ult_c < min_price:
                logger.add(activo, "Pre-Filtro", f"Precio < {min_price}")
                continue
            if ult_v < min_vol:
                logger.add(activo, "Pre-Filtro", f"Volumen < {min_vol}")
                continue
            
            d_base = calcular_indicadores(df_act, bench_s, 20)
            d_base["vix"] = vix_s.reindex(d_base.index, method="ffill")
            mod_res = ejecutar_modelo_multitemporal(d_base, vix_s, logger, activo)
            
            if mod_res is None: continue
            
            bk, inst, expl, fibo = detectores_heuristicos(df_act)
            tags = []
            if bk: tags.append("🚀 Breakout")
            if inst: tags.append("🏦 Inst. Acc.")
            if expl: tags.append("🔥 Momentum")
            if fibo: tags.append("📐 Golden Pocket Fib")

            resultados.append({
                "Activo": activo,
                "Precio": round(float(ult_c), 2),
                "Señal": mod_res["señal"],
                "F(10d)": mod_res["f_10d"],
                "F(20d)": mod_res["f_20d"],
                "F(30d)": mod_res["f_30d"],
                "Fuerza Media": mod_res["fuerza_media"],
                "R² Medio": mod_res["r2_medio"],
                "Win Rate": mod_res["win_rate"],
                "Sharpe": mod_res["sharpe_oos"],
                "Sortino": mod_res["sortino_oos"],
                "Max DD": mod_res["max_dd_oos"],
                "Banderas": " | ".join(tags) if tags else "—"
            })

        barra.empty()
        
        if resultados:
            st.session_state["df_rank"] = pd.DataFrame(resultados).sort_values("Fuerza Media", ascending=False).reset_index(drop=True)
            st.session_state["rank_mercado"] = mercado
            st.session_state["rank_anios"] = anios
            st.success(f"✅ {len(resultados)} activos superaron todos los filtros.")
        st.session_state["df_errores"] = logger.to_df()

    if st.session_state["df_rank"] is not None:
        df_show = st.session_state["df_rank"]
        
        fr1, fr2, fr3 = st.columns(3)
        filtro_s  = fr1.multiselect("Filtrar señal", 
                                    ["COMPRA FUERTE (3/3)", "COMPRAR", "ESPERAR / MIXTO", "VENDER", "VENTA FUERTE (3/3)"], 
                                    default=["COMPRA FUERTE (3/3)", "COMPRAR", "VENTA FUERTE (3/3)", "VENDER"])
        min_r2_rk = fr2.slider("R² promedio mínimo (%)", 0, 20, 1) / 100
        min_wr_rk = fr3.slider("Win Rate histórico mínimo (%)", 0, 100, 60) / 100 

        df_show = df_show[
            df_show["Señal"].isin(filtro_s) & 
            (df_show["R² Medio"] >= min_r2_rk) &
            (df_show["Win Rate"] >= min_wr_rk)
        ]

        def c_senal(v):
            if "COMPRA FUERTE" in v: return "color:#10b981;font-weight:bold"
            if "COMPRAR" in v: return "color:#34d399;font-weight:bold"
            if "VENTA FUERTE" in v: return "color:#ef4444;font-weight:bold"
            if "VENDER" in v: return "color:#f87171;font-weight:bold"
            return "color:#facc15"
        
        st.dataframe(
            df_show.style.map(c_senal, subset=["Señal"]).format({
                "Precio": "${:.2f}",
                "F(10d)": "{:+.2%}",
                "F(20d)": "{:+.2%}",
                "F(30d)": "{:+.2%}",
                "Fuerza Media": "{:+.2%}",
                "R² Medio": "{:.2%}",
                "Win Rate": "{:.1%}",
                "Max DD": "{:.1%}"
            }),
            use_container_width=True, height=480, hide_index=True
        )

        if len(df_show) > 1:
            st.subheader("🗺️ Mapa de Calor de Factores (Z-Score)")
            hm_data = df_show.set_index("Activo")[["Fuerza Media", "R² Medio", "Win Rate", "Sharpe", "Sortino"]].copy()
            fig_heat = px.imshow((hm_data - hm_data.mean()) / hm_data.std().replace(0,1), color_continuous_scale="RdYlGn", aspect="auto")
            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_heat, use_container_width=True)

    if st.session_state["df_errores"] is not None and not st.session_state["df_errores"].empty:
        with st.expander(f"📋 Descartados por Pre-Filtro o Estadística ({len(st.session_state['df_errores'])})", expanded=False):
            st.dataframe(st.session_state["df_errores"], use_container_width=True, hide_index=True)

# ══════════════════════ TAB 6: MI CARTERA EN VIVO (GOOGLE SHEETS) ══════════════════════
with tab6:
    st.markdown("### 💼 Gestión de Cartera en Vivo (Mark-to-Market)")
    st.markdown("Conectado en tiempo real a Google Sheets.")
    
    from streamlit_gsheets import GSheetsConnection
    
    # Establecer conexión con la API de Sheets
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    # Descargar la base de datos
    try:
        df_cartera = conn.read(worksheet="Sheet1", usecols=[0, 1, 2, 3])
        df_cartera = df_cartera.dropna(how="all") # Limpia filas vacías
    except Exception as e:
        st.error(f"Error de lectura en Sheets. Revisá los Secrets de Streamlit y asegurate de que la hoja se llame exactamente 'Sheet1'. Error técnico: {e}")
        df_cartera = pd.DataFrame(columns=["Activo", "Fecha_Compra", "Precio_Compra", "Horizonte_Dias"])

    # 1. Formulario de Ingreso de Operaciones
    with st.expander("➕ Cargar nueva operación", expanded=False):
        with st.form("form_cartera"):
            c_act, c_precio, c_fecha, c_horiz = st.columns(4)
            
            # Consolidar todos los tickers válidos de USA y ARG en una sola lista
            tickers_validos = sorted(list(set(cargar_universo_usa() + cargar_universo_arg())))
            
            # Reemplazamos el text_input por un selectbox bloqueado a los tickers válidos
            n_activo = c_act.selectbox("Ticker", tickers_validos)
            
            n_precio = c_precio.number_input("Precio Compra ($)", min_value=0.01, step=0.5, format="%.2f")
            n_fecha  = c_fecha.date_input("Fecha de Compra")
            n_horiz  = c_horiz.selectbox("Horizonte Objetivo", [10, 20, 30])
            
            submit_btn = st.form_submit_button("Impactar en Google Sheets")
            if submit_btn and n_activo:
                nueva_fila = pd.DataFrame([{
                    "Activo": n_activo,
                    "Fecha_Compra": n_fecha.strftime("%Y-%m-%d"),
                    "Precio_Compra": float(n_precio),
                    "Horizonte_Dias": int(n_horiz)
                }])
                df_actualizado = pd.concat([df_cartera, nueva_fila], ignore_index=True)
                
                # Subir actualización a la nube
                conn.update(worksheet="Sheet1", data=df_actualizado)
                st.success(f"✅ {n_activo} impactado en la base de datos.")
                st.cache_data.clear() # Limpia caché para forzar recarga
                st.rerun()

    # 2. Análisis y Seguimiento de la Cartera
    if not df_cartera.empty:
        st.markdown("#### 📊 Posiciones Activas")
        
        if st.button("🔄 Ejecutar Auditoría de Cartera", type="primary"):
            barra_cartera = st.progress(0, text="Calculando métricas MTM...")
            resultados_cartera = []
            
            hoy_fecha = datetime.today().date()
            bench_cartera = descargar_benchmark(mercado, anios)
            vix_cartera = descargar_vix(anios)
            logger_cartera = ErrorLogger()

            for idx, row in df_cartera.iterrows():
                barra_cartera.progress((idx + 1) / len(df_cartera), text=f"Actualizando {row['Activo']}...")
                
                try:
                    df_act, _, _ = descargar(row["Activo"], 2)
                    if df_act is None: continue
                    
                    precio_actual = float(df_act["Close"].iloc[-1])
                    precio_compra = float(row["Precio_Compra"])
                    rendimiento = (precio_actual / precio_compra) - 1
                    
                    # Motor de Tiempo Hábil
                    fecha_c = datetime.strptime(str(row["Fecha_Compra"]), "%Y-%m-%d").date()
                    dias_transcurridos = np.busday_count(fecha_c, hoy_fecha)
                    dias_restantes = int(row["Horizonte_Dias"]) - dias_transcurridos
                    
                    estado_tiempo = f"⏳ Quedan {dias_restantes}d" if dias_restantes > 0 else "🚨 CERRAR HOY"
                    if dias_restantes < 0:
                        estado_tiempo = f"❌ VENCIDO (Día {dias_transcurridos})"

                    # Re-evaluación algorítmica de la posición en vivo
                    d_base = calcular_indicadores(df_act, bench_cartera, 20)
                    d_base["vix"] = vix_cartera.reindex(d_base.index, method="ffill")
                    mod_res = ejecutar_modelo_multitemporal(d_base, vix_cartera, logger_cartera, row["Activo"])
                    
                    senal_hoy = mod_res["señal"] if mod_res else "RUIDO/DESCARTADO"

                    resultados_cartera.append({
                        "Activo": row["Activo"],
                        "Fecha Compra": row["Fecha_Compra"],
                        "Horizonte": f"{row['Horizonte_Dias']} días",
                        "Precio Compra": round(precio_compra, 2),
                        "Precio Actual": round(precio_actual, 2),
                        "P&L Actual": rendimiento,
                        "Días Restantes": estado_tiempo,
                        "Señal HOY": senal_hoy
                    })
                except Exception:
                    pass
            
            barra_cartera.empty()
            
            if resultados_cartera:
                df_show_cartera = pd.DataFrame(resultados_cartera)
                
                def style_cartera(row_data):
                    style = [''] * len(row_data)
                    idx_pnl = row_data.index.get_loc('P&L Actual')
                    if row_data['P&L Actual'] > 0: style[idx_pnl] = 'color: #34d399; font-weight: bold'
                    elif row_data['P&L Actual'] < 0: style[idx_pnl] = 'color: #f87171; font-weight: bold'
                    
                    idx_dias = row_data.index.get_loc('Días Restantes')
                    if "CERRAR" in row_data['Días Restantes'] or "VENCIDO" in row_data['Días Restantes']:
                        style[idx_dias] = 'background-color: #ef4444; color: white; font-weight: bold'
                        
                    idx_sig = row_data.index.get_loc('Señal HOY')
                    val_sig = str(row_data['Señal HOY'])
                    if "COMPRA" in val_sig: style[idx_sig] = 'color: #34d399'
                    elif "VENTA" in val_sig: style[idx_sig] = 'color: #f87171'
                    else: style[idx_sig] = 'color: #facc15'
                    return style

                st.dataframe(
                    df_show_cartera.style.apply(style_cartera, axis=1).format({
                        "Precio Compra": "${:.2f}",
                        "Precio Actual": "${:.2f}",
                        "P&L Actual": "{:+.2%}"
                    }),
                    use_container_width=True, hide_index=True
                )
                
        if st.button("🗑️ Vaciar Google Sheet (Liquidar Cartera)"):
            df_vacio = pd.DataFrame(columns=["Activo", "Fecha_Compra", "Precio_Compra", "Horizonte_Dias"])
            conn.update(worksheet="Sheet1", data=df_vacio)
            st.cache_data.clear()
            st.rerun()

st.markdown("---")
st.caption(f"Motor: LINEST Walk-Forward Ortogonal · OLS Multitemporal · Golden Pocket Fibonacci · Datos: Yahoo Finance")

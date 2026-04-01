"""
app.py — Modelo IA Screener (USA & ARG)
Motor LINEST Walk-Forward Ortogonal · OLS Multitemporal · Multi-Usuario
Firma: LAUTHARTE · Zoom Estructural · Diagnóstico IA · v6.2 (UI VIX Restaurada)
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
# MÓDULO DE AUTENTICACIÓN MULTI-USUARIO (LOGIN SEGURO)
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
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("## 🔐 Acceso Restringido")
        st.markdown("Plataforma Cuantitativa Institucional. Por favor, identifíquese.")
        
        if "passwords" not in st.secrets:
            st.error("🚨 Error Crítico: Sistema sin configuración de secrets. Contacte al administrador de arquitectura para inyectar credenciales.")
            st.stop()
            
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Usuario", key="username")
            st.text_input("Contraseña", type="password", key="password")
            st.button("Ingresar al Sistema", on_click=password_entered, use_container_width=True)
            if st.session_state.get("password_correct") is False:
                st.error("❌ Usuario o contraseña incorrectos. Verifique sus credenciales.")
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
    sp500, ndx = [], []
    headers = {'User-Agent': 'Mozilla/5.0'}
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
        return sorted(["AAPL","NVDA","TSLA","META","AMZN","MSFT","GOOGL","PLTR","AMD","NFLX","JPM","BAC","V","MA"])
    return universo

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
    c, v = df["Close"], df["Volume"]
    bk = bool(c.iloc[-1] >= c.rolling(50).max().iloc[-2]) if len(c)>50 else False
    ins = bool(v.iloc[-1] > v.rolling(20).mean().iloc[-1] * 2) if len(v)>20 else False
    exp = bool(c.pct_change(20).iloc[-1] > 0.15) if len(c)>20 else False
    return bk, ins, exp

def contexto_vix(vix):
    for n, (lo, hi, f, i, c) in VIX_CONTEXTOS.items():
        if lo <= vix < hi: return n, f, i, c
    return "OPTIMISMO", 1.05, "🔵", "#60a5fa"

# ─────────────────────────────────────────────────────────────────
# IA GENERATIVA RESIDENTE (NLG CUANTITATIVO)
# ─────────────────────────────────────────────────────────────────
def generar_sintesis_quant(ticker, h_data, modelo_res, horizonte, bk, inst, expl, vix, ctx_nom, peso_vix):
    c = h_data.get('Close', 0)
    rsi = h_data.get('rsi', 50)
    macd = h_data.get('macd', 0)
    macd_sig = h_data.get('macd_sig', 0)
    mm10 = h_data.get('mm10', c)
    mm50 = h_data.get('mm50', c)
    
    consenso_pct = modelo_res.get('consenso', 0) * 100
    r2_pct = modelo_res.get('r2_prom', 0) * 100

    tendencia = "alcista primaria" if c > mm50 else "bajista primaria"
    corto_plazo = "acelerando inercia" if c > mm10 else "perdiendo tracción"

    if rsi > 70: rsi_txt = "en zona de sobrecompra técnica (>70), sugiriendo riesgo de corrección inminente"
    elif rsi < 30: rsi_txt = "en zona de sobreventa (<30), indicando posible capitulación y agotamiento vendedor"
    else: rsi_txt = f"en niveles neutrales ({rsi:.1f}), sin extremos tensionales evidentes"

    macd_txt = "cruzado al alza, validando momentum positivo" if macd > macd_sig else "cruzado a la baja, confirmando presión vendedora"

    flags = []
    if bk: flags.append("ruptura de máximos (breakout)")
    if inst: flags.append("acumulación de volumen anormal")
    if expl: flags.append("momentum explosivo de corto plazo")
    
    flags_txt = f" Desde el prisma estructural, se detecta {', '.join(flags)}." if flags else ""
    vix_txt = f"El entorno macro registra volatilidad de {ctx_nom} (VIX: {vix:.2f}), configurado con una ponderación de influencia del {peso_vix}%."

    if consenso_pct > 2.0: veredicto = f"El ensamblaje de regresión dicta postura **COMPRADORA**, proyectando un delta de {consenso_pct:+.2f}% a {horizonte} días."
    elif consenso_pct < -2.0: veredicto = f"El ensamblaje de regresión exige **LIQUIDAR** o mantener postura **VENDEDORA**, proyectando un delta de {consenso_pct:+.2f}% a {horizonte} días."
    else: veredicto = f"El motor dicta postura **NEUTRAL (ESPERAR)**. No se detecta asimetría estadística operable ({consenso_pct:+.2f}% a {horizonte} días)."

    fiabilidad = "alta" if r2_pct >= 1.0 else "marginal (posible ruido)"
    
    texto = f"🤖 **Diagnóstico Algorítmico Integral:**\n\n"
    texto += f"El activo **{ticker}** navega actualmente una tendencia {tendencia}, con el precio de corto plazo {corto_plazo}. "
    texto += f"Mecánicamente, el oscilador RSI opera {rsi_txt}, en confluencia con un MACD {macd_txt}."
    texto += f"{flags_txt} {vix_txt}\n\n"
    texto += f"**Conclusión OLS:** {veredicto} La fiabilidad explicativa de esta lectura es {fiabilidad} (R² Promedio: {r2_pct:.1f}%)."
    
    return texto

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
    yf_target = d["retorno_target"].values
    p1, h1, w1, pw1 = _walk_forward_features(d, FEATS_M1, yf_target, N, ini)
    p2, h2, w2, pw2 = _walk_forward_features(d, FEATS_M2, yf_target, N, ini)
    p3, h3, w3, pw3 = _walk_forward_features(d, FEATS_M3, yf_target, N, ini)
    sum_w = w1+w2+w3
    res = dict(consenso=round((p1*w1+p2*w2+p3*w3)/sum_w, 6) if sum_w>0 else 0.0, 
               r2_prom=round(sum_w/sum(1 for w in [w1,w2,w3] if w>0), 4) if sum_w>0 else 0.0,
               pred_rsi=p1, pred_macd=p2, pred_medias=p3, r2_rsi=w1, r2_macd=w2, r2_medias=w3)
    p_h = pw1 + pw2 + pw3
    with np.errstate(divide="ignore", invalid="ignore"):
        d["consenso_raw"] = np.where(p_h > 0, (np.nan_to_num(h1)*pw1 + np.nan_to_num(h2)*pw2 + np.nan_to_num(h3)*pw3) / p_h, np.nan)
    return res, d

def ejecutar_modelo_multitemporal(d, vix_s, log, tk, peso_vix):
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
    _, cf_base, _, _ = contexto_vix(vh)
    
    cf_adj = 1.0 + (cf_base - 1.0) * (peso_vix / 100.0)
    
    fz, r2 = [], []
    for j in range(3):
        ts = w1[-1,j] + w2[-1,j] + w3[-1,j]
        fz.append(((np.nan_to_num(p1[-1,j])*w1[-1,j] + np.nan_to_num(p2[-1,j])*w2[-1,j] + np.nan_to_num(p3[-1,j])*w3[-1,j])/ts*cf_adj) if ts>0 else 0.0)
        r2.append((ts/sum(1 for w in [w1[-1,j],w2[-1,j],w3[-1,j]] if w>0)) if ts>0 else 0.0)
    
    if max(r2) < R2_MIN: return None
    vc, vv, fm = sum(1 for f in fz if f>0.02), sum(1 for f in fz if f<-0.02), float(np.mean(fz))
    s = "COMPRA FUERTE (3/3)" if vc==3 else ("COMPRAR" if vc>=2 and fm>0.02 else ("VENTA FUERTE (3/3)" if vv==3 else ("VENDER" if vv>=2 and fm<-0.02 else "ESPERAR / MIXTO")))
    
    pw20 = w1[:,1] + w2[:,1] + w3[:,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cons_h20 = np.where(pw20 > 0, (np.nan_to_num(p1[:,1])*w1[:,1] + np.nan_to_num(p2[:,1])*w2[:,1] + np.nan_to_num(p3[:,1])*w3[:,1]) / pw20, np.nan)

    conds_vix = [d["vix"]<15, (d["vix"]>=15)&(d["vix"]<24), (d["vix"]>=24)&(d["vix"]<32), d["vix"]>=32] if "vix" in d.columns else [np.zeros(N,bool)]*4
    base_factors = np.select(conds_vix, [1.0, 1.05, 0.9, 0.75], default=1.0)
    adj_factors = 1.0 + (base_factors - 1.0) * (peso_vix / 100.0)
    cons_f20 = cons_h20 * adj_factors
    sig_raw = np.where(cons_f20 > 0.02, 1, np.where(cons_f20 < -0.02, -1, 0))

    oos_mask = np.arange(N) >= ini
    mask_t = (sig_raw != 0) & np.isfinite(Ym[:,1]) & oos_mask
    
    strat_r = (pd.Series(sig_raw).replace(0, np.nan).ffill(limit=19).fillna(0).shift(1) * d["retorno_diario"].values).dropna()
    met = {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0}
    if len(strat_r) > 5:
        met["sharpe"] = float(np.sqrt(252)*strat_r.mean()/strat_r.std()) if strat_r.std() != 0 else 0.0
        d_neg = strat_r[strat_r < 0]
        met["sortino"] = float(np.sqrt(252)*strat_r.mean()/d_neg.std()) if len(d_neg) > 2 and d_neg.std() != 0 else 0.0
        eq = (1 + strat_r).cumprod()
        met["max_dd"] = float(((eq - eq.cummax()) / eq.cummax()).min())

    wr = float((((sig_raw[mask_t] == 1) & (Ym[:,1][mask_t] > 0)) | ((sig_raw[mask_t] == -1) & (Ym[:,1][mask_t] < 0))).sum() / mask_t.sum()) if mask_t.sum() > 0 else 0.0

    return {"señal": s, "f_10d": round(fz[0], 4), "f_20d": round(fz[1], 4), "f_30d": round(fz[2], 4), "fuerza_media": round(fm, 4), "r2_medio": round(float(np.mean(r2)), 4), "win_rate": wr, "sharpe_oos": round(met["sharpe"], 2), "sortino_oos": round(met["sortino"], 2), "max_dd_oos": round(met["max_dd"], 4)}

def calcular_auditoria_mtm(d, vix_s, h, peso_vix):
    da = d.copy()
    da["vix"] = vix_s.reindex(da.index, method="ffill")
    conds = [da["vix"]<15, (da["vix"]>=15)&(da["vix"]<24), (da["vix"]>=24)&(da["vix"]<32), da["vix"]>=32]
    base_factors = np.select(conds, [1.0, 1.05, 0.9, 0.75], 1.0)
    adj_factors = 1.0 + (base_factors - 1.0) * (peso_vix / 100.0)
    da["consenso_final"] = da["consenso_raw"] * adj_factors
    
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
for k in ["df_rank", "rank_mercado", "rank_anios", "df_errores"]:
    if k not in st.session_state: st.session_state[k] = None

with st.sidebar:
    st.markdown(f"👤 **Usuario Activo:** `{usuario_actual}`")
    if st.button("Cerrar Sesión", use_container_width=True):
        st.session_state["password_correct"] = False
        st.rerun()
    st.markdown("---")
    st.markdown("## ⚙️ Panel de Control")
    mercado = st.radio("Mercado", ["🇺🇸 Estados Unidos", "🇦🇷 Argentina (Merval)"])
    lista = cargar_universo_usa() if "Unidos" in mercado else cargar_universo_arg()
    
    ticker_catalogo = st.selectbox("Catálogo de Índices", lista)
    ticker_manual = st.text_input("🔍 Ticker Libre (Override)", value="", help="Si el activo no está arriba (ej. NU, BABA), escribilo acá.")
    ticker = ticker_manual.upper().strip() if ticker_manual.strip() != "" else ticker_catalogo

    anios = st.slider("Años historia", 2, 10, 3)
    horizonte = st.slider("Horizonte (días)", 5, 60, 20, step=5)
    st.markdown("---")
    peso_vix = st.slider("Ponderación VIX (%)", 0, 100, 100 if "Unidos" in mercado else 30, step=10, help="0% ignora el VIX por completo. 100% aplica el multiplicador institucional.")
    show_bb = st.checkbox("Bandas Bollinger", True)
    show_vol = st.checkbox("Volumen", True)
    show_stoch = st.checkbox("Estocástico %K/%D", False)
    show_atr = st.checkbox("ATR (%)", False)
    st.markdown("---")
    if st.button("🔄 Limpiar caché (Resincronizar BD)", use_container_width=True):
        st.cache_data.clear()
        st.session_state["df_rank"] = None
        st.session_state["df_errores"] = None
        if "df_cartera_cache" in st.session_state: del st.session_state["df_cartera_cache"]
        st.rerun()

st.markdown("# 📊 Modelo IA Screener")
st.markdown(f"`{ticker}` · {mercado} · Objetivo: **{horizonte} días**")
st.markdown("---")

with st.spinner(f"Procesando {ticker}..."):
    df_raw, em, ed = descargar(ticker, anios)
    vix_s = descargar_vix(anios)
    bench_s = descargar_benchmark(mercado, anios)

if df_raw is None: st.error(f"⚠️ {em}"); st.stop()

d = calcular_indicadores(df_raw, bench_s, horizonte)
d["vix"] = vix_s.reindex(d.index, method="ffill")
mod_res, d = ejecutar_modelo(d, horizonte)
bk, ins, exp = detectores_heuristicos(df_raw)

h = d.iloc[-1]
vh = float(h.get("vix", 18.0) or 18.0)
cn, cf_base, ci, _ = contexto_vix(vh)

cf_adj = 1.0 + (cf_base - 1.0) * (peso_vix / 100.0)
cons_adj = mod_res["consenso"] * cf_adj

s_h = "COMPRAR" if cons_adj > 0.02 else ("VENDER" if cons_adj < -0.02 else "ESPERAR")
sc = {"COMPRAR":"#34d399", "VENDER":"#f87171", "ESPERAR":"#facc15"}[s_h]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(f"<div style='background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;text-align:center'><div style='font-size:11px;color:#64748b;text-transform:uppercase'>Señal {horizonte}d</div><div style='font-size:1.7rem;font-weight:700;color:{sc}'>{s_h}</div><div style='font-size:12px;color:#64748b'>{'+' if cons_adj >= 0 else ''}{cons_adj*100:.2f}%</div></div>", unsafe_allow_html=True)
c2.metric("Precio", f"${h['Close']:.2f}")

# --- Corrección Visual VIX ---
c3.metric(f"VIX {ci}", f"{vh:.2f}", f"{cn} (Ajuste {peso_vix}%)", delta_color="off")
# -----------------------------

c4.metric("RSI-14", f"{h['rsi']:.1f}", "Sobrecompra" if h["rsi"] > 70 else ("Sobreventa" if h["rsi"] < 30 else "Normal"), delta_color="inverse" if h["rsi"] > 70 else "normal")
c5.metric("MACD", f"{h['macd']:.4f}", "↑ Alcista" if h["macd"] > h["macd_sig"] else "↓ Bajista", delta_color="normal" if h["macd"] > h["macd_sig"] else "inverse")
c6.metric("R² Prom", f"{mod_res['r2_prom']*100:.1f}%", "Significativo" if mod_res["r2_prom"] >= R2_MIN else "Sin señal")

if any([bk, ins, exp]):
    tags = [t for t, c in zip(["🚀 Breakout (Máx 50d)", "🏦 Acum. Inst. (Vol. 2x)", "🔥 Momentum (>15%)"], [bk, ins, exp]) if c]
    st.markdown(f"**Banderas activas:** `{'` · `'.join(tags)}`")

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Gráfico", "🧠 Modelo LINEST", "🕵️ Auditoría OOS", "📋 Datos", "🏆 Ranking Global", "💼 Mi Cartera"])

# ══════════════════════ TAB 1: GRÁFICO ══════════════════════
with tab1:
    rows_n = 1 + sum([show_vol, show_stoch, show_atr])
    h_rows = [0.55] + [0.15] * (rows_n - 1)
    subs = ["Precio"] + (["Volumen"] if show_vol else []) + (["Estocástico"] if show_stoch else []) + (["ATR %"] if show_atr else [])
    fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=h_rows, subplot_titles=subs)
    
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Precio", increasing_line_color="#34d399", decreasing_line_color="#f87171", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["mm10"], name="MM10", line=dict(color="#facc15", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["mm50"], name="MM50", line=dict(color="#f87171", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema12"], name="EMA12", line=dict(color="#a78bfa", width=1, dash="dash"), visible="legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema26"], name="EMA26", line=dict(color="#818cf8", width=1, dash="dash"), visible="legendonly"), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], name="BB+", line=dict(color="rgba(148,163,184,0.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], name="BB-", fill="tonexty", fillcolor="rgba(148,163,184,0.07)", line=dict(color="rgba(148,163,184,0.4)", width=1)), row=1, col=1)
    if cons_adj != 0.0 and mod_res["r2_prom"]>=R2_MIN:
        obj = h["Close"]*(1+cons_adj)
        fig.add_hline(y=obj, line_dash="dash", line_color=sc, annotation_text=f"Objetivo: ${obj:.2f}", row=1, col=1)

    curr = 2
    if show_vol:
        clrs = ["#34d399" if cl>=op else "#f87171" for cl, op in zip(d["Close"], d["Open"])]
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=clrs, showlegend=False), row=curr, col=1)
        fig.update_yaxes(fixedrange=True, row=curr, col=1); curr += 1
    if show_stoch and "stoch_k" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["stoch_k"], name="%K", line=dict(color="#60a5fa", width=1.5)), row=curr, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["stoch_d"], name="%D", line=dict(color="#f472b6", width=1.5, dash="dot")), row=curr, col=1)
        fig.update_yaxes(fixedrange=True, row=curr, col=1); curr += 1
    if show_atr and "atr_pct" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["atr_pct"]*100, name="ATR %", fill="tozeroy", fillcolor="rgba(167,139,250,0.15)", line=dict(color="#a78bfa", width=1.5)), row=curr, col=1)
        fig.update_yaxes(fixedrange=True, row=curr, col=1)

    fig.update_layout(height=600 + (rows_n-1)*130, xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    txt_sintesis = generar_sintesis_quant(ticker, h, mod_res, horizonte, bk, ins, exp, vh, cn, peso_vix)
    st.info(txt_sintesis)

# ══════════════════════ TAB 2: MODELO ══════════════════════
with tab2:
    st.markdown(f"### 🧠 Resultados LINEST Walk-Forward ({horizonte}d)")
    c = st.columns(3)
    for i, (n, p, r, ds) in enumerate(zip(["🔵 M1: Momentum", "🟣 M2: Divergencia", "🟡 M3: Tendencia"], 
                              [mod_res["pred_rsi"], mod_res["pred_macd"], mod_res["pred_medias"]], 
                              [mod_res["r2_rsi"], mod_res["r2_macd"], mod_res["r2_medias"]], 
                              ["RSI · ATR · F.Rel · Ret1d · Ret3d", "var. MACD · ATR · F.Rel · Ret5d · Gap OC", "desv. MM50 · MM10 vs MM50 · var. Vol · Gap OC"])):
        with c[i]:
            st.markdown(f"#### {n}")
            act = r >= R2_MIN
            st.metric("Predicción", f"{p*100:+.2f}%" if act else "— filtrado")
            st.metric("R² adj (peso)", f"{r*100:.2f}%", "✅ Significativo" if act else "❌ Ruido — descartado", delta_color="normal" if act else "inverse")
            st.caption(ds)
            
    st.markdown("---")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Consenso ponderado", f"{mod_res['consenso']*100:+.2f}%")
    cc2.metric(f"Factor VIX ({ci} {cn})", f"{cf_adj:.2f}x", f"VIX {vh:.1f} | Pond: {peso_vix}%", delta_color="off")
    cc3.metric("Consenso ajustado VIX", f"{cons_adj*100:+.2f}%", f"→ {s_h}", delta_color="normal" if cons_adj > 0 else "inverse")

# ══════════════════════ TAB 3: AUDITORÍA OOS ══════════════════════
with tab3:
    st.markdown(f"### 🕵️ Auditoría Out-Of-Sample (Horizonte: {horizonte}d)")
    tr, da, met = calcular_auditoria_mtm(d, vix_s, horizonte, peso_vix)
    if tr.empty: st.info("Sin señales históricas útiles.")
    else:
        ac1, ac2, ac3, ac4, ac5, ac6, ac7 = st.columns(7)
        ac1.metric("Operaciones", len(tr)); ac2.metric("Aciertos", (tr["resultado"] == "✅ ACIERTO").sum()); ac3.metric("Fallos", len(tr) - (tr["resultado"] == "✅ ACIERTO").sum())
        ac4.metric("Win Rate", f"{met['win_rate']*100:.1f}%", "Bueno" if met['win_rate']>=0.60 else "Peligroso", delta_color="normal" if met['win_rate']>=0.60 else "inverse")
        ac5.metric("Sharpe MTM", f"{met['sharpe']:.2f}"); ac6.metric("Sortino MTM", f"{met['sortino']:.2f}"); ac7.metric("Max Drawdown", f"{met['max_dd']*100:.1f}%", delta_color="inverse")
        if "equity_curve" in da.columns:
            fig_eq = px.line(da, x=da.index, y="equity_curve", title="Curva de Capital (MTM Diario)")
            fig_eq.add_hline(y=1, line_dash="dash", line_color="#94a3b8")
            fig_eq.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_eq, use_container_width=True)
        st.dataframe(tr.rename(columns={"Close":"Precio","consenso_final":"Consenso", "señal_h":"Señal","retorno_target":f"Ret {horizonte}d","resultado":"Resultado"})
            .style.format({"Precio":"${:.2f}","vix":"{:.1f}","rsi":"{:.1f}", "Consenso":"{:+.3f}",f"Ret {horizonte}d":"{:+.2%}"}), use_container_width=True, height=280)

# ══════════════════════ TAB 4: DATOS ══════════════════════
with tab4:
    st.markdown("### 📋 Últimas 100 filas de datos")
    st.dataframe(d[[c for c in ["Close", "Volume", "rsi", "macd", "mm10", "mm50", "atr_pct", "fuerza_rel", "vix", "consenso_raw"] if c in d.columns]].tail(100).sort_index(ascending=False).style.format("{:.4f}"), use_container_width=True, height=450)

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
                logger.add(activo, err_m, err_d); continue
                
            ult_c = df_act["Close"].iloc[-1]
            ult_v = df_act["Volume"].iloc[-20:].mean()
            
            if ult_c < min_price:
                logger.add(activo, "Pre-Filtro", f"Precio < {min_price}"); continue
            if ult_v < min_vol:
                logger.add(activo, "Pre-Filtro", f"Volumen < {min_vol}"); continue
            
            d_base = calcular_indicadores(df_act, bench_s, 20)
            d_base["vix"] = vix_s.reindex(d_base.index, method="ffill")
            mod_r = ejecutar_modelo_multitemporal(d_base, vix_s, logger, activo, peso_vix)
            
            if mod_r is None: continue
            
            bk_a, inst_a, expl_a = detectores_heuristicos(df_act)
            tags = []
            if bk_a: tags.append("🚀 Breakout")
            if inst_a: tags.append("🏦 Inst. Acc.")
            if expl_a: tags.append("🔥 Momentum")

            resultados.append({
                "Activo": activo, "Precio": round(float(ult_c), 2), "Señal": mod_r["señal"],
                "F(10d)": mod_r["f_10d"], "F(20d)": mod_r["f_20d"], "F(30d)": mod_r["f_30d"],
                "Fuerza Media": mod_r["fuerza_media"], "R² Medio": mod_r["r2_medio"], "Win Rate": mod_r["win_rate"],
                "Sharpe": mod_r["sharpe_oos"], "Sortino": mod_r["sortino_oos"], "Max DD": mod_r["max_dd_oos"],
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
        filtro_s  = fr1.multiselect("Filtrar señal", ["COMPRA FUERTE (3/3)", "COMPRAR", "ESPERAR / MIXTO", "VENDER", "VENTA FUERTE (3/3)"], default=["COMPRA FUERTE (3/3)", "COMPRAR", "VENTA FUERTE (3/3)", "VENDER"])
        min_r2_rk = fr2.slider("R² promedio mínimo (%)", 0, 20, 1) / 100
        min_wr_rk = fr3.slider("Win Rate histórico mínimo (%)", 0, 100, 60) / 100 

        df_show = df_show[df_show["Señal"].isin(filtro_s) & (df_show["R² Medio"] >= min_r2_rk) & (df_show["Win Rate"] >= min_wr_rk)]

        def c_senal(v):
            if "COMPRA FUERTE" in v: return "color:#10b981;font-weight:bold"
            if "COMPRAR" in v: return "color:#34d399;font-weight:bold"
            if "VENTA FUERTE" in v: return "color:#ef4444;font-weight:bold"
            if "VENDER" in v: return "color:#f87171;font-weight:bold"
            return "color:#facc15"
        
        st.dataframe(df_show.style.map(c_senal, subset=["Señal"]).format({"Precio": "${:.2f}", "F(10d)": "{:+.2%}", "F(20d)": "{:+.2%}", "F(30d)": "{:+.2%}", "Fuerza Media": "{:+.2%}", "R² Medio": "{:.2%}", "Win Rate": "{:.1%}", "Max DD": "{:.1%}"}), use_container_width=True, height=480, hide_index=True)

        if len(df_show) > 1:
            st.subheader("🗺️ Mapa de Calor de Factores (Z-Score)")
            hm_data = df_show.set_index("Activo")[["Fuerza Media", "R² Medio", "Win Rate", "Sharpe", "Sortino"]].copy()
            fig_heat = px.imshow((hm_data - hm_data.mean()) / hm_data.std().replace(0,1), color_continuous_scale="RdYlGn", aspect="auto")
            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_heat, use_container_width=True)

    if st.session_state["df_errores"] is not None and not st.session_state["df_errores"].empty:
        with st.expander(f"📋 Descartados por Pre-Filtro o Estadística ({len(st.session_state['df_errores'])})", expanded=False):
            st.dataframe(st.session_state["df_errores"], use_container_width=True, hide_index=True)

# ══════════════════════ TAB 6: MI CARTERA EN VIVO ══════════════════════
with tab6:
    st.markdown("### 💼 Gestión de Cartera Multi-Usuario (Google Sheets)")
    st.markdown(f"Bienvenido/a, **{usuario_actual}**. Este es tu portafolio personal. Las operaciones que ingreses aquí no se mezclarán con las de otros usuarios del sistema.")
    
    from streamlit_gsheets import GSheetsConnection
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    if "df_cartera_cache" not in st.session_state:
        try:
            df_bd = conn.read(worksheet="Sheet1")
            st.session_state["df_cartera_cache"] = df_bd.dropna(how="all") 
        except Exception as e:
            st.error(f"Error de lectura en Sheets. Revisá los Secrets. Error: {e}")
            st.session_state["df_cartera_cache"] = pd.DataFrame(columns=["Usuario", "Activo", "Fecha_Compra", "Precio_Compra", "Horizonte_Dias", "Estado", "Fecha_Cierre", "Precio_Cierre", "Resultado_Pct"])

    df_completo = st.session_state["df_cartera_cache"]

    columnas_esperadas = ["Usuario", "Activo", "Fecha_Compra", "Precio_Compra", "Horizonte_Dias", "Estado", "Fecha_Cierre", "Precio_Cierre", "Resultado_Pct"]
    for col in columnas_esperadas:
        if col not in df_completo.columns: df_completo[col] = None
            
    columnas_texto = ["Usuario", "Activo", "Fecha_Compra", "Estado", "Fecha_Cierre"]
    for col in columnas_texto:
        df_completo[col] = df_completo[col].astype(object)

    df_completo["Estado"] = df_completo["Estado"].fillna("ABIERTA")
    df_completo["Usuario"] = df_completo["Usuario"].fillna("admin")
    df_completo["Precio_Compra"] = pd.to_numeric(df_completo["Precio_Compra"], errors="coerce")
    df_completo["Resultado_Pct"] = pd.to_numeric(df_completo["Resultado_Pct"], errors="coerce")
    
    df_cartera = df_completo[df_completo["Usuario"] == usuario_actual].copy()

    df_abiertas = df_cartera[df_cartera["Estado"] == "ABIERTA"].copy()
    df_cerradas = df_cartera[df_cartera["Estado"] == "CERRADA"].copy()

    with st.expander("➕ Abrir Nueva Posición", expanded=False):
        with st.form("form_cartera"):
            c_act, c_precio, c_fecha, c_horiz = st.columns(4)
            tickers_validos = sorted(list(set(cargar_universo_usa() + cargar_universo_arg())))
            n_activo = c_act.selectbox("Ticker", tickers_validos)
            n_precio = c_precio.number_input("Precio Compra ($)", min_value=0.01, step=0.5, format="%.2f")
            n_fecha  = c_fecha.date_input("Fecha de Compra")
            n_horiz  = c_horiz.selectbox("Horizonte Objetivo", [10, 20, 30])
            
            if st.form_submit_button("Impactar en Google Sheets") and n_activo:
                nueva_fila = pd.DataFrame([{"Usuario": usuario_actual, "Activo": n_activo, "Fecha_Compra": n_fecha.strftime("%Y-%m-%d"), "Precio_Compra": float(n_precio), "Horizonte_Dias": int(n_horiz), "Estado": "ABIERTA", "Fecha_Cierre": None, "Precio_Cierre": None, "Resultado_Pct": None}])
                df_actualizado = pd.concat([df_completo, nueva_fila], ignore_index=True)
                conn.update(worksheet="Sheet1", data=df_actualizado)
                del st.session_state["df_cartera_cache"] 
                st.success(f"✅ {n_activo} comprada e impactada para {usuario_actual}.")
                st.rerun()

    if not df_abiertas.empty:
        st.markdown("#### 📊 Posiciones Activas")
        
        if st.button("🔄 Ejecutar Auditoría en Vivo", type="primary"):
            barra_cartera = st.progress(0, text="Calculando métricas MTM...")
            resultados_cartera = []
            hoy_fecha = datetime.today().date()
            bench_cartera = descargar_benchmark(mercado, anios)
            vix_cartera = descargar_vix(anios)
            logger_cartera = ErrorLogger()

            for idx, row in df_abiertas.iterrows():
                barra_cartera.progress((list(df_abiertas.index).index(idx) + 1) / len(df_abiertas), text=f"Actualizando {row['Activo']}...")
                try:
                    df_act, _, _ = descargar(row["Activo"], 2)
                    if df_act is None: continue
                    precio_actual = float(df_act["Close"].iloc[-1])
                    precio_compra = float(row["Precio_Compra"])
                    rendimiento = (precio_actual / precio_compra) - 1
                    
                    fecha_c = pd.to_datetime(row["Fecha_Compra"]).date()
                    dias_transcurridos = np.busday_count(fecha_c, hoy_fecha)
                    dias_restantes = int(row["Horizonte_Dias"]) - dias_transcurridos
                    
                    estado_tiempo = f"⏳ Quedan {dias_restantes}d" if dias_restantes > 0 else "🚨 CERRAR HOY"
                    if dias_restantes < 0: estado_tiempo = f"❌ VENCIDO (Día {dias_transcurridos})"

                    d_base = calcular_indicadores(df_act, bench_cartera, 20)
                    d_base["vix"] = vix_cartera.reindex(d_base.index, method="ffill")
                    mod_r2 = ejecutar_modelo_multitemporal(d_base, vix_cartera, logger_cartera, row["Activo"], peso_vix)
                    senal_hoy = mod_r2["señal"] if mod_r2 else "RUIDO/DESCARTADO"

                    resultados_cartera.append({"Activo": row["Activo"], "Fecha Compra": row["Fecha_Compra"], "Horizonte": f"{row['Horizonte_Dias']} días", "Precio Compra": round(precio_compra, 2), "Precio Actual": round(precio_actual, 2), "P&L Actual": rendimiento, "Días Restantes": estado_tiempo, "Señal HOY": senal_hoy})
                except Exception as e: 
                    st.warning(f"Error procesando {row['Activo']}: {str(e)[:50]}") 
            
            barra_cartera.empty()
            if resultados_cartera:
                df_show_cartera = pd.DataFrame(resultados_cartera)
                def style_cartera(row_data):
                    style = [''] * len(row_data)
                    ipnl = row_data.index.get_loc('P&L Actual')
                    if row_data['P&L Actual'] > 0: style[ipnl] = 'color: #34d399; font-weight: bold'
                    elif row_data['P&L Actual'] < 0: style[ipnl] = 'color: #f87171; font-weight: bold'
                    
                    idias = row_data.index.get_loc('Días Restantes')
                    if "CERRAR" in row_data['Días Restantes'] or "VENCIDO" in row_data['Días Restantes']: style[idias] = 'background-color: #ef4444; color: white; font-weight: bold'
                        
                    isig = row_data.index.get_loc('Señal HOY')
                    vsig = str(row_data['Señal HOY'])
                    if "COMPRA" in vsig: style[isig] = 'color: #34d399'
                    elif "VENTA" in vsig: style[isig] = 'color: #f87171'
                    else: style[isig] = 'color: #facc15'
                    return style

                st.dataframe(df_show_cartera.style.apply(style_cartera, axis=1).format({"Precio Compra": "${:.2f}", "Precio Actual": "${:.2f}", "P&L Actual": "{:+.2%}"}), use_container_width=True, hide_index=True)

        st.markdown("#### ❌ Cerrar Posición")
        with st.form("form_cierre"):
            cl1, cl2, cl3 = st.columns(3)
            df_abiertas["Label_Cierre"] = df_abiertas.apply(lambda x: f"{x['Activo']} (C: {x['Fecha_Compra']} a ${x['Precio_Compra']:.2f})", axis=1)
            label_dict = dict(zip(df_abiertas["Label_Cierre"], df_abiertas.index))
            
            ticker_cierre_lbl = cl1.selectbox("Seleccionar operación a liquidar", list(label_dict.keys()))
            precio_cierre = cl2.number_input("Precio de Venta / Cierre ($)", min_value=0.01, step=0.5, format="%.2f")
            
            if st.form_submit_button("Liquidar Operación") and ticker_cierre_lbl:
                idx_to_close = label_dict[ticker_cierre_lbl]
                precio_compra_original = float(df_completo.at[idx_to_close, "Precio_Compra"])
                resultado_final = (precio_cierre / precio_compra_original) - 1
                
                df_completo.at[idx_to_close, "Estado"] = "CERRADA"
                df_completo.at[idx_to_close, "Fecha_Cierre"] = datetime.today().strftime("%Y-%m-%d")
                df_completo.at[idx_to_close, "Precio_Cierre"] = precio_cierre
                df_completo.at[idx_to_close, "Resultado_Pct"] = resultado_final
                
                conn.update(worksheet="Sheet1", data=df_completo)
                del st.session_state["df_cartera_cache"] 
                st.success(f"✅ Operación cerrada exitosamente. P&L: {resultado_final*100:+.2f}%.")
                st.rerun()
    else:
        st.info("📌 No tenés posiciones activas en este momento. Usá el formulario superior para abrir una.")

    st.markdown("---")
    if not df_cerradas.empty:
        st.markdown("#### 📜 Historial de Operaciones (Track Record)")
        aciertos = (df_cerradas["Resultado_Pct"] > 0).sum()
        total_cerradas = len(df_cerradas)
        win_rate = aciertos / total_cerradas
        acumulado_pct = df_cerradas["Resultado_Pct"].sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Operaciones Cerradas", total_cerradas)
        m2.metric("Win Rate Histórico", f"{win_rate*100:.1f}%")
        m3.metric("P&L Acumulado", f"{acumulado_pct*100:+.2f}%", delta_color="normal" if acumulado_pct > 0 else "inverse")
        
        df_show_hist = df_cerradas[["Activo", "Fecha_Compra", "Precio_Compra", "Fecha_Cierre", "Precio_Cierre", "Resultado_Pct"]].copy()
        
        def style_historial(val):
            if isinstance(val, float):
                if val > 0 and val < 1: return 'color: #34d399; font-weight: bold'
                elif val < 0: return 'color: #f87171; font-weight: bold'
            return ''

        st.dataframe(df_show_hist.style.map(style_historial, subset=['Resultado_Pct']).format({"Precio_Compra": "${:.2f}", "Precio_Cierre": "${:.2f}", "Resultado_Pct": "{:+.2%}"}), use_container_width=True, hide_index=True)
    else:
        st.markdown("#### 📜 Historial de Operaciones (Track Record)")
        st.info("📉 Aún no tenés operaciones cerradas en tu historial. Liquida una posición activa para empezar a registrar tu Win Rate y P&L acumulado.")

    with st.expander("⚠️ Zona de Peligro (Precaución)", expanded=False):
        st.error("Atención: Vaciar la base de datos elimina permanentemente todas TUS posiciones abiertas y todo TU historial. No afectará a otros usuarios.")
        if st.button("🗑️ Vaciar Mi Cartera Completamente"):
            df_restante = df_completo[df_completo["Usuario"] != usuario_actual]
            conn.update(worksheet="Sheet1", data=df_restante)
            if "df_cartera_cache" in st.session_state: del st.session_state["df_cartera_cache"]
            st.rerun()

# ─────────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("**Modelo IA Screener v6.2** | Desarrollado por: **LAUTHARTE**")
st.caption("⚠️ **Aviso Legal:** Este sistema es una herramienta de análisis cuantitativo creada exclusivamente con fines educativos e informativos. NO constituye asesoramiento financiero, de inversión, legal ni fiscal. Los resultados históricos de la auditoría OOS no garantizan rendimientos futuros. Las señales del modelo son estimaciones estadísticas con incertidumbre. El uso de este sistema es bajo su propio riesgo y responsabilidad.")

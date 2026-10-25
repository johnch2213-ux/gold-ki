import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Gold-KI Dashboard", page_icon="🥇", layout="wide")
GRAMS_PER_TROY_OUNCE = 31.1034768
HORIZONS = {"3 Monate": 63, "6 Monate": 126, "12 Monate": 252}

st.title("🥇 Gold-KI-Dashboard")
st.caption("Goldpreis in Euro pro Gramm · Prognosen · Risiko · Altgold-Rechner")
st.warning("Forschungsprototyp, keine Anlageberatung. GC=F sind Gold-Futures und kein perfekter Spotpreis. Prognosen sind unsicher; historische Szenarien sind keine garantierten Prognoseintervalle.")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(years):
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    def close(ticker, name):
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"Keine Daten für {ticker}.")
        s = raw["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.rename(name)
    df = pd.concat([close("GC=F", "gold_usd_oz"), close("EURUSD=X", "eurusd")], axis=1).sort_index().ffill().dropna()
    df["gold_eur_g"] = df["gold_usd_oz"] / df["eurusd"] / GRAMS_PER_TROY_OUNCE
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def make_features(data):
    price = data["gold_eur_g"]
    logp = np.log(price)
    frame = pd.DataFrame(index=data.index)
    for lag in [1, 2, 3, 5, 10, 21, 42, 63, 126, 252]:
        frame[f"ret_{lag}"] = logp.diff(lag)
    for window in [5, 21, 63, 126, 252]:
        frame[f"vol_{window}"] = logp.diff().rolling(window).std() * np.sqrt(252)
        frame[f"mom_{window}"] = logp.diff(window)
    for window in [21, 63, 126, 252]:
        frame[f"price_vs_ma_{window}"] = price / price.rolling(window).mean() - 1
    frame["gold_usd_return_21"] = np.log(data["gold_usd_oz"]).diff(21)
    frame["eurusd_return_21"] = np.log(data["eurusd"]).diff(21)
    return frame.replace([np.inf, -np.inf], np.nan)

def build_forecasts(data):
    price = data["gold_eur_g"].dropna()
    features = make_features(data)
    latest_x = features.dropna().iloc[[-1]]
    last_price, last_date = float(price.iloc[-1]), price.index[-1]
    forecast_rows, backtest_rows = [], []
    for label, h in HORIZONS.items():
        target = np.log(price.shift(-h) / price).rename("target_return")
        ds = features.join(target).dropna()
        X, y = ds.drop(columns=["target_return"]), ds["target_return"]
        if len(ds) < 500:
            raise RuntimeError(f"Zu wenig Daten für einen stabilen Modelltest ({len(ds)} Datensätze).")
        split = int(len(ds) * .8)
        model = HistGradientBoostingRegressor(max_iter=180, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
        model.fit(X.iloc[:split], y.iloc[:split])
        pred = model.predict(X.iloc[split:])
        model_mae = mean_absolute_error(y.iloc[split:], pred)
        base_mae = mean_absolute_error(y.iloc[split:], np.zeros(len(y.iloc[split:])))
        backtest_rows.append({"Horizont": label, "Testpunkte": len(y.iloc[split:]), "MAE Modell": model_mae, "MAE Baseline": base_mae, "Besser als Baseline": model_mae < base_mae})
        final_model = HistGradientBoostingRegressor(max_iter=180, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
        final_model.fit(X, y)
        pred_return = float(final_model.predict(latest_x)[0])
        pred_price = last_price * np.exp(pred_return)
        realized = np.log(price.shift(-h) / price).dropna()
        lo, hi = realized.quantile([.10, .90])
        forecast_rows.append({"Horizont": label, "Handelstage": h, "Prognose €/g": pred_price, "Änderung %": (pred_price / last_price - 1) * 100, "Historisches 10%-Szenario €/g": last_price * np.exp(lo), "Historisches 90%-Szenario €/g": last_price * np.exp(hi), "Modell-MAE": model_mae, "Baseline-MAE": base_mae})
    return last_price, last_date, pd.DataFrame(forecast_rows), pd.DataFrame(backtest_rows)

with st.sidebar:
    st.header("Einstellungen")
    years = st.selectbox("Historische Daten", [10, 15, 20], index=2)
    st.caption("Bis zu 20 Jahre, soweit verfügbar.")
    if st.button("Daten neu laden"):
        load_data.clear()
        st.rerun()

try:
    with st.spinner("Lade Marktdaten und trainiere Modelle …"):
        data = load_data(years)
        last_price, last_date, forecast, backtest = build_forecasts(data)
except Exception as e:
    st.error(f"Daten oder Modell konnten nicht geladen werden: {e}")
    st.info("Bitte später erneut versuchen. Die externe Datenquelle kann zeitweise nicht erreichbar sein.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Goldwert (modelliert)", f"{last_price:.2f} €/g", f"Datenstand: {last_date:%d.%m.%Y}")
m2.metric("Gold-Futures", f"{float(data['gold_usd_oz'].iloc[-1]):,.0f} USD/oz")
m3.metric("EUR/USD", f"{float(data['eurusd'].iloc[-1]):.4f}")

st.subheader("KI-Prognosen")
cols = st.columns(3)
for col, (_, row) in zip(cols, forecast.iterrows()):
    col.metric(row["Horizont"], f"{row['Prognose €/g']:.2f} €/g", f"{row['Änderung %']:+.2f}%")
    signal = "🟢 Eher positiv" if row["Änderung %"] >= 3 else ("🔴 Eher negativ" if row["Änderung %"] <= -3 else "🟡 Neutral")
    col.write(f"**Modell-Signal:** {signal}")
    col.caption(f"Historisches 10.–90%-Szenario: {row['Historisches 10%-Szenario €/g']:.2f}–{row['Historisches 90%-Szenario €/g']:.2f} €/g")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["gold_eur_g"], name="Historischer Goldwert (€/g)", mode="lines"))
for _, row in forecast.iterrows():
    future_date = last_date + pd.tseries.offsets.BDay(int(row["Handelstage"]))
    fig.add_trace(go.Scatter(x=[last_date, future_date], y=[last_price, row["Prognose €/g"]], mode="lines+markers", name=f"Prognose {row['Horizont']}", line=dict(dash="dash")))
fig.update_layout(title="Historischer Goldwert und Modellprognosen", xaxis_title="Datum", yaxis_title="Euro pro Gramm", hovermode="x unified", height=520)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Backtest: Modell gegen einfache Baseline")
st.dataframe(backtest.style.format({"MAE Modell":"{:.4f}", "MAE Baseline":"{:.4f}"}), use_container_width=True)
if not backtest["Besser als Baseline"].all():
    st.warning("Das Modell schlägt die einfache 0%-Rendite-Baseline nicht bei jedem Horizont. Prognosen besonders vorsichtig interpretieren.")
st.caption("Erster zeitlich geordneter Holdout-Test, kein vollständiger Walk-forward-Backtest. Bei langen Horizonten können sich Zielzeiträume überlappen.")

st.subheader("Altgold-Rechner")
c1, c2, c3 = st.columns(3)
weight = c1.number_input("Gewicht in Gramm", min_value=0.0, value=10.0, step=.5)
purity_label = c2.selectbox("Legierung", ["999er / Feingold", "916er / 22 Karat", "750er / 18 Karat", "585er / 14 Karat", "375er / 9 Karat"], index=3)
discount = c3.slider("Beispielhafter Händlerabschlag (%)", 0, 30, 10)
purities = {"999er / Feingold":.999, "916er / 22 Karat":.916, "750er / 18 Karat":.750, "585er / 14 Karat":.585, "375er / 9 Karat":.375}
gross = weight * purities[purity_label] * last_price
offer = gross * (1 - discount / 100)
a, b, c = st.columns(3)
a.metric("Enthaltenes Feingold", f"{weight * purities[purity_label]:.3f} g")
b.metric("Rechnerischer Metallwert", f"{gross:.2f} €")
c.metric("Beispiel nach Abschlag", f"{offer:.2f} €")
st.caption("Der Abschlag ist frei einstellbar und kein garantierter Händlerkurs. Der tatsächliche Ankaufspreis hängt vom Händler und der Prüfung ab.")

st.download_button("Prognosen als CSV herunterladen", forecast.to_csv(index=False).encode("utf-8"), "gold_prognosen.csv", "text/csv")
st.download_button("Backtest als CSV herunterladen", backtest.to_csv(index=False).encode("utf-8"), "gold_backtest.csv", "text/csv")
st.caption("Quelle: Yahoo Finance (GC=F und EURUSD=X). Gold-Futures-basierter Näherungswert, nicht zwingend offizieller Spot- oder Händlerpreis.")

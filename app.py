import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Gold-KI Pro", page_icon="🥇", layout="wide")
GRAMS_PER_TROY_OUNCE = 31.1034768
HORIZONS = {"3 Monate": 63, "6 Monate": 126, "12 Monate": 252}
PURITY = {
    "999er / Feingold": .999, "916er / 22 Karat": .916,
    "750er / 18 Karat": .750, "585er / 14 Karat": .585,
    "375er / 9 Karat": .375, "333er / 8 Karat": .333,
}

st.title("🥇 Gold-KI Pro")
st.caption("Goldwert in €/g · 3/6/12-Monats-Szenarien · Modellprüfung · Altgold-Rechner")
st.warning(
    "Forschungsprototyp, keine Anlageberatung. Verwendet werden Gold-Futures (GC=F) und EUR/USD von Yahoo Finance "
    "als Näherung, nicht zwingend der offizielle Spot- oder Händlerpreis. Szenarien sind historische Vergleichswerte, "
    "keine garantierten Prognoseintervalle."
)

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

    df = pd.concat([close("GC=F", "gold_usd_oz"), close("EURUSD=X", "eurusd")], axis=1)
    df = df.sort_index().ffill().dropna()
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

def new_model():
    return HistGradientBoostingRegressor(
        max_iter=180, learning_rate=.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42
    )

def build_forecasts(data):
    price = data["gold_eur_g"].dropna()
    features = make_features(data)
    latest_x = features.dropna().iloc[[-1]]
    last_price, last_date = float(price.iloc[-1]), price.index[-1]
    forecast_rows, backtest_rows = [], []

    for label, h in HORIZONS.items():
        target = np.log(price.shift(-h) / price).rename("target_return")
        ds = features.join(target).dropna()
        if len(ds) < max(500, h * 3):
            raise RuntimeError(f"Zu wenig Daten für den Horizont {label} ({len(ds)} Datensätze).")

        # Zeitlich geordneter Holdout mit Purge-Gap: Trainingslabels reichen nicht in den Testzeitraum hinein.
        split = int(len(ds) * .8)
        train_end = max(1, split - h)
        X, y = ds.drop(columns=["target_return"]), ds["target_return"]
        if train_end < 200 or split >= len(ds) - 20:
            raise RuntimeError("Zu wenige Daten für einen sinnvollen zeitlichen Modelltest.")

        test_x, test_y = X.iloc[split:], y.iloc[split:]
        test_model = new_model()
        test_model.fit(X.iloc[:train_end], y.iloc[:train_end])
        pred = test_model.predict(test_x)
        model_mae = mean_absolute_error(test_y, pred)
        base_mae = mean_absolute_error(test_y, np.zeros(len(test_y)))
        directional = float(np.mean(np.sign(pred) == np.sign(test_y)) * 100)
        backtest_rows.append({
            "Horizont": label, "Testpunkte": len(test_y),
            "MAE Modell (Log-Rendite)": model_mae,
            "MAE 0%-Baseline": base_mae,
            "Richtung richtig (%)": directional,
            "Besser als Baseline": model_mae < base_mae,
        })

        final_model = new_model()
        final_model.fit(X, y)
        pred_return = float(final_model.predict(latest_x)[0])
        pred_price = last_price * np.exp(pred_return)

        # Historische realisierte Forward-Renditen als anschauliche Szenarioverteilung.
        realized = target.dropna()
        q10, q50, q90 = realized.quantile([.10, .50, .90])
        forecast_rows.append({
            "Horizont": label, "Handelstage": h,
            "Prognose €/g": pred_price,
            "Änderung %": (pred_price / last_price - 1) * 100,
            "Pessimistisches historisches Szenario €/g": last_price * np.exp(q10),
            "Median historisches Szenario €/g": last_price * np.exp(q50),
            "Optimistisches historisches Szenario €/g": last_price * np.exp(q90),
            "Modell-MAE": model_mae, "Baseline-MAE": base_mae,
        })
    return last_price, last_date, pd.DataFrame(forecast_rows), pd.DataFrame(backtest_rows)

with st.sidebar:
    st.header("Einstellungen")
    years = st.selectbox("Historischer Zeitraum", [10, 15, 20], index=2)
    st.caption("Bis zu 20 Jahre, soweit die Datenquelle verfügbar ist.")
    if st.button("Daten neu laden"):
        load_data.clear()
        st.rerun()

try:
    with st.spinner("Lade Marktdaten und trainiere die Modelle …"):
        data = load_data(years)
        last_price, last_date, forecast, backtest = build_forecasts(data)
except Exception as e:
    st.error(f"Daten oder Modell konnten nicht geladen werden: {e}")
    st.info("Bitte später erneut versuchen. Die externe Datenquelle kann zeitweise nicht erreichbar sein.")
    st.stop()

st.subheader("Marktüberblick")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Gold-Näherungswert", f"{last_price:.2f} €/g", f"Stand {last_date:%d.%m.%Y}")
m2.metric("Gold-Futures", f"{float(data['gold_usd_oz'].iloc[-1]):,.0f} USD/oz")
m3.metric("EUR/USD", f"{float(data['eurusd'].iloc[-1]):.4f}")
m4.metric("Historische Daten", f"{len(data):,} Handelstage")

st.subheader("KI-Prognosen und Szenarien")
st.caption("Die Punktprognose stammt aus dem Modell. Optimistisch/pessimistisch zeigen das 10.–90. Perzentil historischer Vorwärtsrenditen desselben Horizonts – keine statistisch kalibrierten Konfidenzgrenzen.")
cols = st.columns(3)
for col, (_, row) in zip(cols, forecast.iterrows()):
    change = row["Änderung %"]
    col.metric(row["Horizont"], f"{row['Prognose €/g']:.2f} €/g", f"{change:+.2f}% zur aktuellen Näherung")
    col.write(f"🔴 **Pessimistisch:** {row['Pessimistisches historisches Szenario €/g']:.2f} €/g")
    col.write(f"🟡 **Historischer Median:** {row['Median historisches Szenario €/g']:.2f} €/g")
    col.write(f"🟢 **Optimistisch:** {row['Optimistisches historisches Szenario €/g']:.2f} €/g")
    col.caption("Modellrichtung: " + ("eher positiv" if change >= 3 else "eher negativ" if change <= -3 else "neutral"))

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["gold_eur_g"], name="Historischer Goldwert (€/g)", mode="lines"))
for _, row in forecast.iterrows():
    future_date = last_date + pd.tseries.offsets.BDay(int(row["Handelstage"]))
    fig.add_trace(go.Scatter(
        x=[last_date, future_date], y=[last_price, row["Prognose €/g"]],
        mode="lines+markers", name=f"KI {row['Horizont']}", line=dict(dash="dash")
    ))
fig.update_layout(
    title="Historischer Goldwert und KI-Punktprognosen",
    xaxis_title="Datum", yaxis_title="Euro pro Gramm",
    hovermode="x unified", height=520, legend_title="Reihe"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Modellprüfung")
st.dataframe(
    backtest.style.format({
        "MAE Modell (Log-Rendite)": "{:.4f}",
        "MAE 0%-Baseline": "{:.4f}",
        "Richtung richtig (%)": "{:.1f}",
    }),
    use_container_width=True
)
if not backtest["Besser als Baseline"].all():
    st.warning("Das Modell schlägt die einfache 0%-Rendite-Baseline nicht bei jedem Horizont. Die Punktprognose daher besonders vorsichtig interpretieren.")
st.caption("Der Holdout ist zeitlich geordnet und enthält einen Purge-Abstand in Höhe des Prognosehorizonts, um Überschneidungen von Trainingszielen mit dem Testzeitraum zu verringern. Das ist weiterhin kein vollständiger rollierender Walk-forward-Test.")

st.subheader("Altgold-Ankaufsschätzer")
c1, c2, c3 = st.columns(3)
weight = c1.number_input("Gewicht (Gramm)", min_value=0.0, value=10.0, step=.5)
purity_label = c2.selectbox("Legierung", list(PURITY.keys()), index=3)
discount = c3.slider("Beispielhafter Händlerabschlag (%)", 0, 35, 10)
fine_gold = weight * PURITY[purity_label]
gross = fine_gold * last_price
offer = gross * (1 - discount / 100)
a, b, c = st.columns(3)
a.metric("Geschätztes Feingold", f"{fine_gold:.3f} g")
b.metric("Rechnerischer Metallwert", f"{gross:.2f} €")
c.metric("Beispiel nach Abschlag", f"{offer:.2f} €")
st.caption("Der Abschlag ist ein frei wählbarer Rechenwert, keine Marktquote oder Zusage. Steine, Verschmutzung, Schmelzverluste und Händlerbedingungen können den tatsächlichen Preis verändern.")

st.subheader("Datenexport")
d1, d2 = st.columns(2)
d1.download_button("Prognosen als CSV", forecast.to_csv(index=False).encode("utf-8-sig"), "gold_prognosen.csv", "text/csv")
d2.download_button("Modellprüfung als CSV", backtest.to_csv(index=False).encode("utf-8-sig"), "gold_backtest.csv", "text/csv")
st.caption("Datenquelle: Yahoo Finance (GC=F, EURUSD=X). Daten werden bis zu eine Stunde zwischengespeichert. Keine Anlageberatung.")

# gold-ki
# Gold-KI-Dashboard (Web-App)

## Online bereitstellen
1. Lege ein GitHub-Konto und ein Repository an.
2. Lade `app.py` und `requirements.txt` in das Repository hoch.
3. Öffne https://share.streamlit.io/ und verbinde GitHub.
4. Wähle das Repository, Branch `main` und Hauptdatei `app.py`.
5. Klicke auf Deploy. Verfügbarkeit und Limits richten sich nach den aktuellen Bedingungen des Hosters.

## Einschränkungen
- Yahoo Finance; `GC=F` ist Gold-Futures, kein perfekter Spotpreis.
- Experimenteller Prototyp, keine Anlageberatung.
- Historische 10.–90%-Szenarien sind keine kalibrierten Prognoseintervalle.
- Der Holdout-Test ist kein vollständiger Walk-forward-Backtest.

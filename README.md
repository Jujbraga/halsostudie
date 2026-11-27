# halsostudie

Uppgift i Pythonprogrammering och statistisk dataanalys om en Hälsostudie

## Innehåll

- Beskrivande analys och diagram
- Simulering av andel sjuka (numpy)
- Konfidensintervall för systoliskt blodtryck (t-fördelning och bootstrap)
- Hypotesprövning: har rökare högre medelblodtryck än icke-rökare? (Welchs t-test)
- Power-simulering för hypotestestet
- Enkel linjär regression: blodtryck utifrån ålder
- Multipel linjär regression: blodtryck utifrån ålder och vikt, med residualplot

## Python version

- Python 3.14.7

## Projektstruktur

```
halsostudie/
├── report.ipynb                # Notebook med analys och slutsatser
├── health_study_dataset.csv    # Datamängd (800 personer)
├── requirements.txt            # Paket och versioner
├── README.md
└── src/
    ├── analyzer.py             # Klassen HealthAnalyzer
    ├── functions.py            # Statistik, simuleringar, hypotesprövning och regression
    └── viz.py                  # Diagramfunktioner
```

## Installation

1. Klona repot.
2. Skapa och aktivera en virtuell miljö:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

3. Installera paketen:

```bash
pip install -r requirements.txt
```

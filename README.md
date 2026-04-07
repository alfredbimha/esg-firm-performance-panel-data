# Green vs Brown Stock Volatility (GARCH)

## Research Question
Do clean energy stocks exhibit different volatility dynamics than fossil fuel stocks?

## Methodology
**Language:** Python  
**Methods:** GARCH(1,1), EGARCH, GJR-GARCH

## Data
Yahoo Finance — ICLN, XLE, SPY (2015–2025)

## Key Findings
High volatility persistence across all assets; asymmetric leverage effects differ between green and brown stocks; COVID-19 caused divergent recovery patterns.

## How to Run
```bash
pip install -r requirements.txt
python code/project1_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*

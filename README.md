# ESG Scores and Firm Financial Performance (Panel Data)

## Research Question
Is there a relationship between ESG performance and firm financial returns?

## Methodology
**Language:** Python  
**Methods:** Pooled OLS, Fixed Effects, Two-way FE

## Data
Yahoo Finance (30 firms × 7 years), sector ESG benchmarks

## Key Findings
ESG-return relationship varies by estimation method; two-way FE shows significant positive association.

## How to Run
```bash
pip install -r requirements.txt
python code/project2_*.py
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

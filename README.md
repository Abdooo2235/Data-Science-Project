# Data Science Project

Main project: **[Stock Market Trend Analysis](Stock-Market-Trend-Analysis/)** — forecast next-day
price movement for the S&P 500 and three tech stocks (AAPL, AMZN, NVDA), comparing classical
(ARIMA/GARCH), tree (LightGBM), and deep (LSTM) models. Honest result: **no usable out-of-sample
price skill** (efficient-market ceiling), but **volatility is predictable**.

> Educational only — **not investment advice.**

---

## Run in Google Colab (easiest)

### 1. New Colab notebook → paste this in the first cell → run it

```python
# Get the project files (the project is a subfolder, so we move it into place)
!rm -rf /content/dsp /content/Stock-Market-Trend-Analysis
!git clone https://github.com/Abdooo2235/Data-Science-Project.git /content/dsp
!mv /content/dsp/Stock-Market-Trend-Analysis /content/Stock-Market-Trend-Analysis
!pip install -q -r /content/Stock-Market-Trend-Analysis/requirements.txt
print("Ready. Now open a notebook from notebooks/ and Run all.")
```

### 2. Open the notebook you want

`File → Open notebook → GitHub` → search **`Abdooo2235/Data-Science-Project`** → pick one:

| Notebook | What it does |
|---|---|
| `notebooks/01_data_collection_preprocessing.ipynb` | Fetch + clean + engineer features (builds the data) |
| `notebooks/02_eda.ipynb` | Exploratory data analysis (18 figures) |
| `notebooks/03_model_building.ipynb` | ARIMA + GARCH + LightGBM + LSTM |
| `notebooks/04_evaluation_presentation.ipynb` | Final evaluation |
| `notebooks/04_volatility_modeling.ipynb` | Volatility forecasting |
| `notebooks/05_volatility_regime.ipynb` | Calm-vs-stormy regime classifier |

### 3. `Runtime → Run all`

That's it.

**Notes**
- The processed data ships with the repo, so **02/03/04/05 run on their own** — you don't have to run 01 first.
- Repo is public — no login or token needed.
- For the LSTM in notebook 03, pick a GPU: `Runtime → Change runtime type → T4 GPU`.

---

## Run locally (Python 3.10+)

```bash
git clone https://github.com/Abdooo2235/Data-Science-Project.git
cd Data-Science-Project/Stock-Market-Trend-Analysis
py -3.10 -m venv .venv && .venv\Scripts\activate      # Windows
python -m pip install -r requirements.txt
python notebooks/01_data_collection_preprocessing.py   # then 02, 03, ...
```

Full architecture, decisions, and milestone reports: **[Stock-Market-Trend-Analysis/README.md](Stock-Market-Trend-Analysis/README.md)**.

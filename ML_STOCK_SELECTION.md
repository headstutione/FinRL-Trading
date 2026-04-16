# ML Stock Selection Pipeline

End-to-end guide for running ML-driven stock selection with per-sector-bucket models.

---

## Overview

```
Step 1: Setup              →  .env, venv, dependencies
Step 2: Fetch Data         →  FMP API → SQLite + CSV
Step 2b: Backfill History  →  Fill ex-SP500 members (survivorship-bias-free)
Step 3: Data Cleaning      →  Fix adj_close_q, fill y_return, remove bad records
Step 4: Run ML             →  4 bucket models → predictions CSV
Step 5: Mixed-Vintage Run  →  Use latest available data per ticker for today's picks
```

**Output:** Ranked stock picks per bucket (growth_tech, cyclical, real_assets, defensive) with 7 competing ML models.

---

## Prerequisites

### 1. Environment Variables

Create `.env` in project root (copy from `.env.example`):

```bash
# Required: FMP API key (https://financialmodelingprep.com/)
FMP_API_KEY=your_fmp_api_key_here
```

### 2. Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

# Core dependencies
pip install pandas numpy scikit-learn requests pyyaml pandas-market-calendars tzlocal tqdm

# Optional but recommended (better ML models)
pip install lightgbm xgboost

# macOS only: LightGBM requires libomp
brew install libomp
```

---

## Step 1: Fetch & Store Fundamental Data

**Script:** `src/data/fetch_and_store_fundamentals.py`

```bash
source venv/bin/activate

# Fetch all SP500 stocks, 2021-2026, save to DB + CSV
python3 src/data/fetch_and_store_fundamentals.py \
    --start-date 2021-01-01 \
    --end-date 2026-04-01 \
    --output-csv data/fundamentals_full.csv
```

**Arguments:**

| Arg | Default | Description |
|-----|---------|-------------|
| `--start-date` | 2021-01-01 | Fundamental data start date |
| `--end-date` | 2026-04-01 | Fundamental data end date |
| `--limit` | 10000 | Max number of tickers to fetch |
| `--preferred-source` | FMP | Data source (FMP recommended) |
| `--output-csv` | None | Optional CSV output path |
| `--survivorship-free` | off | Use all historical SP500 tickers (~755) instead of current ~503 to avoid survivorship bias |

**Survivorship-free mode:**

```bash
# Fetch fundamentals for ALL tickers that have ever been in SP500 since 2015
python3 src/data/fetch_and_store_fundamentals.py \
    --survivorship-free \
    --start-date 2015-01-01 \
    --end-date 2026-04-01
```

When `--survivorship-free` is enabled, the script reads `data/sp500_historical_constituents.csv` to collect every ticker that has appeared in SP500 since `--start-date` (~755 tickers since 2015 vs ~503 current). This includes ~252 tickers that were removed from the index (e.g., due to acquisitions, delistings, downsizing). Already-delisted tickers that fail to fetch are safely skipped.

**What it does:**

1. Fetches SP500 universe via FMP API (`/sp500-constituent`)
2. For each ticker, calls 4 FMP endpoints:
   - `/income-statement?period=quarter&limit=40`
   - `/balance-sheet-statement?period=quarter&limit=40`
   - `/cash-flow-statement?period=quarter&limit=40`
   - `/ratios?period=quarter&limit=40`
3. Computes 52 fundamental factors per quarter (extracted from FMP ratios endpoint via `RATIO_FIELD_MAP`):

   | Category | Factors |
   |----------|---------|
   | Valuation (7) | PE, PS, PB, PEG, price_to_fcf, price_to_ocf, ev_multiple |
   | Profitability (9) | EPS, ROE, net_income_ratio, gross_margin, operating_margin, ebitda_margin, pretax_margin, effective_tax_rate, ebt_per_ebit |
   | Liquidity (3) | cur_ratio, quick_ratio, cash_ratio |
   | Leverage (8) | debt_ratio, debt_to_equity, debt_to_assets, debt_to_capital, lt_debt_to_capital, interest_coverage, debt_service_coverage, debt_to_mktcap |
   | Efficiency (6) | acc_rec_turnover, asset_turnover, fixed_asset_turnover, inventory_turnover, payables_turnover, wc_turnover |
   | Cash Flow (10) | fcf_per_share, ocf_per_share, cash_per_share, capex_per_share, fcf_to_ocf, ocf_ratio, ocf_to_sales, ocf_coverage, st_ocf_coverage, capex_coverage |
   | Per-Share (5) | BPS, DPS, revenue_per_share, tangible_bvps, interest_debt_per_share |
   | Dividend (3) | dividend_payout, dividend_yield, div_capex_coverage |
   | Solvency (1) | solvency_ratio |

   **Removed (collinear duplicates):** `price_to_fair_value` (= PB), `financial_leverage` (= debt_to_equity + 1), `net_income_per_ebt` (= 1 - effective_tax_rate)

   **Data sources:**
   - `DPS`: from `ratio_q['dividendPerShare']`, fallback `cash_q['commonDividendsPaid'] / shares`
   - `net_income_ratio`: from `ratio_q['netProfitMargin']` (not income statement)

4. Computes forward log return (`y_return`):
   ```
   y_return[t] = ln(adj_close_q[t+1] / adj_close_q[t])
   ```
   - For the latest quarter lacking next-quarter fundamentals, uses **price data** to fill y_return
   - e.g., Q4 2025 y_return uses 2026-03-31 closing price even if Q1 2026 financials aren't published

5. Stores to SQLite (`data/finrl_trading.db`, table `fundamental_data`) and optional CSV

**Output:** ~25,000 records (714 tickers x ~37 quarters with survivorship-free)

**Key code:** `src/data/data_fetcher.py`
- `FMPFetcher.get_fundamental_data()` (line 548) — main fetch + factor computation
- `FMPFetcher._fetch_fmp_data()` (line 226) — API call with local-first caching
- y_return computation (line 872) — forward log return with price-based fallback

**Key code:** `src/data/data_store.py`
- `save_fundamental_data()` — upsert to SQLite fundamental_data table
- `get_fundamental_data()` — query with optional ticker/date filters

---

## Step 1b: Backfill Historical SP500 Members

**Script:** `src/data/backfill_historical_sp500.py`

Fills gaps for ex-SP500 members (mergers, delistings, name changes) that were never fetched. Uses `sp500_historical_constituents.csv` to identify missing tickers per quarter (2016-Q1 ~ 2025-Q1).

```bash
source venv/bin/activate
python3 src/data/backfill_historical_sp500.py
```

**What it does:**

1. Identifies missing (ticker, quarter) pairs by comparing SP500 membership vs DB records
2. Fetches fundamentals from FMP for missing tickers (~322 unique)
3. Fills tradedate, actual_tradedate, trade_price for new records
4. Recomputes y_return for affected tickers
5. Verifies per-quarter coverage (target: >=480/505)

**Tradedate mapping:**

| datadate (quarter end) | tradedate (first day of "next-next" month) |
|------------------------|---------------------------------------------|
| 12-31 | 03-01 (next year) |
| 03-31 | 06-01 |
| 06-30 | 09-01 |
| 09-30 | 12-01 |

---

## Step 1c: Fix adj_close_q and y_return

**Scripts:**
- `src/data/fix_adj_close.py` — Fix frozen/stale adj_close_q using yfinance prices, then recompute all y_return
- `src/data/fill_recent_yreturn.py` — Fill y_return for the latest 1-2 quarters using current prices

```bash
# Fix all adj_close_q with fresh yfinance data + recompute y_return
python3 src/data/fix_adj_close.py

# Fill Q4 2025 and Q1 2026 y_return with today's prices
python3 src/data/fill_recent_yreturn.py
```

---

## Step 2: Verify Data Quality

Before running ML, verify the data:

```bash
source venv/bin/activate
python3 -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/finrl_trading.db')

# Check overall stats
df = pd.read_sql('SELECT datadate, COUNT(*) as cnt, SUM(CASE WHEN y_return IS NOT NULL THEN 1 ELSE 0 END) as has_y FROM fundamental_data GROUP BY datadate ORDER BY datadate', conn)
print(df.to_string(index=False))

# Check latest quarter y_return coverage
latest = df.iloc[-1]['datadate']
missing = pd.read_sql(f\"SELECT COUNT(*) as missing FROM fundamental_data WHERE datadate='{latest}' AND y_return IS NULL\", conn)
print(f'\nLatest quarter ({latest}): {missing.iloc[0][\"missing\"]} missing y_return')
conn.close()
"
```

**Expected:** Each quarter should have ~500 records. The latest quarter (inference target) should have y_return = NaN (this is what ML predicts).

---

## Step 3: Fill Missing y_return (if needed)

If the latest-but-one quarter has missing y_return (e.g., Q4 2025 missing because Q1 2026 fundamentals aren't published for most stocks, but prices ARE available):

```bash
source venv/bin/activate
python3 -c "
import sqlite3, pandas as pd, numpy as np
import sys; sys.path.insert(0, 'src')
from data.data_fetcher import FMPFetcher

conn = sqlite3.connect('data/finrl_trading.db')

# Find rows missing y_return for the quarter you need
TARGET_DATE = '2025-12-31'  # adjust as needed
q = pd.read_sql(f\"SELECT id, ticker, adj_close_q, y_return FROM fundamental_data WHERE datadate='{TARGET_DATE}'\", conn)
missing = q[q['y_return'].isna() | (q['y_return'] == 0)]
print(f'{TARGET_DATE}: {len(missing)} missing y_return out of {len(q)}')

if len(missing) > 0:
    f = FMPFetcher()
    tickers_df = pd.DataFrame({'tickers': missing['ticker'].unique().tolist()})
    prices = f.get_price_data(tickers_df, '2026-03-24', '2026-04-05')  # next quarter end
    prices['datadate'] = pd.to_datetime(prices['datadate'], errors='coerce')
    target = pd.Timestamp('2026-03-31')  # next quarter end date

    updates = []
    for _, row in missing.iterrows():
        tp = prices[prices['tic'] == row['ticker']].sort_values('datadate')
        if tp.empty or row['adj_close_q'] <= 0: continue
        tp_near = tp.iloc[(tp['datadate'] - target).abs().argsort()[:1]]
        next_price = tp_near.iloc[0]['adj_close']
        if pd.notna(next_price) and float(next_price) > 0:
            updates.append((np.log(float(next_price) / row['adj_close_q']), row['id']))

    cur = conn.cursor()
    cur.executemany('UPDATE fundamental_data SET y_return = ? WHERE id = ?', updates)
    conn.commit()
    print(f'Updated {len(updates)} rows')
conn.close()
"
```

---

## Step 4: Run ML Bucket Selection

**Script:** `src/strategies/ml_bucket_selection.py`

```bash
source venv/bin/activate

# Default run
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-09-30 --survivorship-free

# Mixed-vintage: use latest available data per ticker (Q1 early reporters + Q4 rest)
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-09-30 --survivorship-free --mixed-vintage
```

**Arguments:**

| Arg | Default | Description |
|-----|---------|-------------|
| `--db` | data/finrl_trading.db | SQLite database path |
| `--val-cutoff` | 2025-12-31 | Last quarter whose y_return is fully realized |
| `--val-quarters` | 3 | Number of validation quarters |
| `--output-dir` | data/ | Output directory for CSVs |
| `--survivorship-free` | off | Filter training data by historical SP500 membership per quarter |
| `--mixed-vintage` | off | Inference: per current-SP500 ticker, use latest available datadate |

**What it does:**

### 4a. Sector-to-Bucket Mapping

All SP500 stocks are classified into 4 buckets by GICS sector:

```
growth_tech:  Information Technology, Communication Services
cyclical:     Consumer Discretionary, Financials, Industrials
real_assets:  Energy, Materials, Real Estate
defensive:    Healthcare, Consumer Staples, Utilities
```

**Key code:** `src/strategies/group_selection_by_gics.py` — `SECTOR_TO_BUCKET` dict (line 41)

### 4b. Feature Preprocessing

1. Fill missing values with global median per feature
2. Replace `inf`/`-inf` with 0
3. **Winsorize**: clip all features at 1st/99th percentile to reduce outlier impact (e.g., `debt_service_coverage` max 6,837 → clipped to p99)
4. StandardScaler before model training

### 4c. Survivorship-Free Filtering (optional)

When `--survivorship-free` is enabled, the pipeline uses `data/sp500_historical_constituents.csv` to filter training/validation data:

- For each quarter ≤ val_cutoff: only keep tickers that were **actually in SP500 at that date** (lookup the nearest snapshot ≤ quarter date)
- For inference quarters (> val_cutoff): **no filtering** — uses all available tickers

This eliminates survivorship bias where stocks that were later removed from the index (bankruptcies, acquisitions, delistings) are excluded from historical training data, making backtests look artificially better.

```bash
# Run with survivorship-free filtering
python3 src/strategies/ml_bucket_selection.py --survivorship-free
```

### 4d. Train / Validation / Inference Split

**Critical: `--val-cutoff` must be the last quarter whose `y_return` is fully realized.**

`y_return` = `log(next_quarter_trade_price / this_quarter_trade_price)`. For this to be "fully realized", **both** trade dates must be in the past.

```
datadate    tradedate    y_return = log(P_next / P_this)         status (as of 2026-04-17)
─────────   ─────────    ────────────────────────────────         ────────────────────────
2025-06-30  2025-09-01   log(P_2025-12-01 / P_2025-09-01)       fully realized ✓ → train/val
2025-09-30  2025-12-01   log(P_2026-03-01 / P_2025-12-01)       fully realized ✓ → val (last)
2025-12-31  2026-03-01   log(P_2026-06-01 / P_2026-03-01)       6/1 NOT YET ✗  → inference
2026-03-31  2026-06-01   log(P_2026-09-01 / P_2026-06-01)       6/1 NOT YET ✗  → inference
```

**Therefore `--val-cutoff 2025-09-30`** is the correct setting as of April 2026.

> Note: DB may have y_return filled for 2025-12-31 using today's price as a proxy, but this is a partial (~1.5 month) return, not the real quarterly return. Using it for training would introduce noise.

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAIN: 39 quarters (2015-Q2 ~ 2024-Q4)                        │
│  ~620 records/quarter, survivorship-free filtered               │
│  y_return fully realized ✓                                      │
├─────────────────────────────────────────────────────────────────┤
│  VALIDATION: 3 quarters (2025-Q1, Q2, Q3)                      │
│  Used for model selection (best MSE wins)                       │
│  After selection, retrain on train+val combined                 │
├─────────────────────────────────────────────────────────────────┤
│  INFERENCE: datadate > val_cutoff                               │
│  Default: 2025-12-31 (~610) + 2026-03-31 (~96)                 │
│  Mixed-vintage: latest per SP500 ticker → 499 stocks            │
└─────────────────────────────────────────────────────────────────┘
```

With `--survivorship-free`, train/validation sets are further filtered to only include tickers that were SP500 members at each respective quarter.

### 4e. Mixed-Vintage Mode (`--mixed-vintage`)

In practice, not all companies report at the same time. As of mid-April 2026:
- ~76 companies already reported Q1 2026 earnings (datadate=2026-03-31)
- ~423 companies only have Q4 2025 data (datadate=2025-12-31)

`--mixed-vintage` uses the **latest available** data per current-SP500 ticker:

```
  499 current SP500 tickers
    ├── 76 early reporters → use Q1 2026 (datadate=2026-03-31)
    └── 423 not yet reported → use Q4 2025 (datadate=2025-12-31)
```

All 499 tickers are ranked together within each bucket (not separately by datadate). Output CSV includes a `data_vintage` column (Q1_2026 or Q4_2025) and `original_datadate`.

```bash
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-09-30 --survivorship-free --mixed-vintage
```

### 4f. Model Competition (7 models per bucket)

| # | Model | Library | Key Params |
|---|-------|---------|------------|
| 1 | RandomForest | sklearn | n_estimators=200, max_depth=8 |
| 2 | XGBoost | xgboost | n_estimators=200, max_depth=6, lr=0.05 |
| 3 | LightGBM | lightgbm | n_estimators=200, max_depth=6, lr=0.05 |
| 4 | HistGradientBoosting | sklearn | max_iter=200, max_depth=6, lr=0.05 |
| 5 | ExtraTrees | sklearn | n_estimators=200, max_depth=8 |
| 6 | Ridge | sklearn | alpha=1.0 (linear baseline) |
| 7 | Stacking | sklearn | Top 3 models + Ridge meta-learner, cv=3 |

- Best model selected by **validation MSE** (lowest wins)
- Each bucket may have a different best model

### 4g. Ensemble Prediction

All 7 models generate predictions. The ensemble is an **inverse-MSE weighted average**:

```
weight_i = (1 / MSE_i) / sum(1 / MSE_j for all j)
pred_ensemble = sum(weight_i * pred_i)
```

Models with lower MSE get higher weight in the ensemble.

### 4h. Ranking

Within each bucket, stocks are ranked by:
- `rank_best` — ranking by the best model's predicted return
- `rank_ensemble` — ranking by the MSE-weighted ensemble prediction

---

## Output Files

### `data/ml_bucket_predictions.csv`

One row per stock in the inference set:

| Column | Description |
|--------|-------------|
| tic | Ticker symbol |
| datadate | Quarter date (e.g., 2026-03-31) |
| gsector | GICS sector |
| bucket | growth_tech / cyclical / real_assets / defensive |
| adj_close_q | Quarter-end adjusted close price |
| EPS...solvency_ratio | 52 fundamental factor values |
| y_return | Actual forward return (NaN if future) |
| best_model | Name of best model for this bucket |
| predicted_return | Best model's predicted return |
| pred_RF | RandomForest prediction |
| pred_XGB | XGBoost prediction |
| pred_LGBM | LightGBM prediction |
| pred_HistGBM | HistGradientBoosting prediction |
| pred_ExtraTrees | ExtraTrees prediction |
| pred_Ridge | Ridge prediction |
| pred_Stacking | Stacking ensemble prediction |
| pred_ensemble_avg | Inverse-MSE weighted ensemble prediction |
| rank_best | Within-bucket rank by best model |
| rank_ensemble | Within-bucket rank by ensemble |

### `data/ml_bucket_model_results.csv`

One row per model per bucket:

| Column | Description |
|--------|-------------|
| bucket | Bucket name |
| model | Model name |
| val_mse | Validation MSE |
| train_size | Number of training samples |
| val_size | Number of validation samples |
| infer_size | Number of inference samples |

### `data/ml_feature_importance.csv`

One row per feature per model per bucket (all models that expose feature importance):

| Column | Description |
|--------|-------------|
| bucket | Bucket name |
| model | Model name (RF, XGB, LGBM, HistGBM, ExtraTrees, Ridge) |
| is_best | Whether this is the best model for the bucket |
| feature | Factor name |
| importance | Importance value (tree: impurity-based; Ridge: normalized abs coef) |
| rank | 1 = most important |

---

## Example: Full Run

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Fetch fundamental data (survivorship-free: ALL historical SP500 tickers)
python3 src/data/fetch_and_store_fundamentals.py \
    --survivorship-free \
    --start-date 2015-01-01 \
    --end-date 2026-04-01

# 3. Backfill ex-SP500 members
python3 src/data/backfill_historical_sp500.py

# 4. Fix adj_close_q and fill recent y_return
python3 src/data/fix_adj_close.py
python3 src/data/fill_recent_yreturn.py

# 5. Run ML selection (mixed-vintage for today's picks)
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-09-30 --survivorship-free --mixed-vintage

# 6. View results
python3 -c "
import pandas as pd
df = pd.read_csv('data/sp500_ml_bucket_predictions_*.csv')
for bucket in df['bucket'].unique():
    b = df[df['bucket']==bucket].sort_values('rank_best')
    print(f'\n{bucket.upper()} (best model: {b.iloc[0][\"best_model\"]}):')
    print(b[['tic','predicted_return','rank_best','rank_ensemble']].head(5).to_string(index=False))
"
```

### Backtesting a Specific Quarter

```bash
# Backtest 2025/12/1 → 2026/3/1 (Q4 2025 trade period)
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-06-30 --survivorship-free

# Backtest 2026/3/1 → today
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-09-30 --survivorship-free
```

For backtesting, set `--val-cutoff` to the datadate **one quarter before** the inference period starts:
- Trade 12/1 → val-cutoff = 2025-06-30 (inference includes 2025-09-30 → tradedate 12/1)
- Trade 3/1 → val-cutoff = 2025-09-30 (inference includes 2025-12-31 → tradedate 3/1)

---

## Codebase Reference

```
src/data/
  data_fetcher.py                   # FMP API client, factor computation, y_return
                                    #   get_sp500_members_at_date() — lookup historical SP500 members
                                    #   get_all_historical_sp500_tickers() — all tickers ever in SP500
  data_store.py                     # SQLite persistence layer
  fetch_and_store_fundamentals.py   # Step 1: Fetch data from FMP → SQLite (--survivorship-free)
  backfill_historical_sp500.py      # Step 1b: Backfill ex-SP500 members for survivorship-free
  fix_adj_close.py                  # Fix adj_close_q via yfinance + recompute y_return
  fill_recent_yreturn.py            # Fill y_return for latest quarters with current prices

src/strategies/
  group_selection_by_gics.py        # SECTOR_TO_BUCKET mapping, YAML update logic
  ml_strategy.py                    # Underlying ML training engine (rolling/single mode)
  ml_bucket_selection.py            # Step 4: Train & predict per bucket
                                    #   --survivorship-free, --mixed-vintage
  AdaptiveRotationConf_v1.2.2.yaml  # Strategy config (4 buckets)

data/
  finrl_trading.db                  # SQLite database (fundamental_data table)
                                    #   ~25,600 records, 714 tickers, 64 columns
  sp500_historical_constituents.csv # Historical SP500 membership (1996–2026, 2705 rows)
  sp500_ml_bucket_predictions_*.csv # ML predictions (timestamped)
  sp500_ml_bucket_model_results_*.csv
  sp500_ml_feature_importance_*.csv
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Only 5 quarters of data | FMP default `limit=5` | Already fixed: `limit=40` in `_fetch_fmp_data` |
| Same old data after key change | Raw payload cache in `data/cache/finrl_trading.db` | Clear: `DELETE FROM raw_payloads` in the **cache** DB |
| Q4 2025 y_return all NaN | No Q1 2026 fundamental rows for most stocks | Run Step 3 (fill using price data) |
| LightGBM import error | Missing libomp on macOS | `brew install libomp` |
| "训练或交易样本不足" | Not enough training quarters | Use `--test-quarters 1` or fetch more history |
| real_assets bucket skipped | No stocks in that bucket have latest quarter data | Wait for earnings season |

---

## Data Quality Cleanup (performed once)

The following data issues were identified and cleaned:

| Issue | Records | Fix |
|-------|---------|-----|
| Pre-IPO fake records (ABNB, COIN, CRWD etc. before their IPO dates with NULL trade_price) | 209 deleted | Delete records before actual IPO |
| Trailing NULL records for delisted/acquired tickers | 258 deleted | Delete last record(s) where trade_price=NULL |
| Leading NULL trade_price (pre-listing records) | 114 deleted | Delete records before first valid trade_price |
| Tickers with all-zero trade_price (AABA, BXLT, CE etc.) | 152 deleted | Delete entire ticker |
| Extreme outlier records (CVNA, STI, VRT, FTR adj_close_q/trade_price mismatch) | 4 deleted | Delete + recompute y_return for neighbors |
| NULL trade_price for delisted tickers (last record) | 161 deleted | Delete unfixable records |
| Empty columns (net_income_per_ebt, financial_leverage, price_to_fair_value) | 3 columns | Drop columns entirely |
| NULL gsector | 115 tickers | Fill via FMP `/profile` API + manual mapping for 27 delisted tickers |
| NULL y_return for delisted tickers | 93 filled | Use FMP historical-price-eod for last available price |

**After cleanup:** 25,617 records, 714 tickers, 64 columns, NULL y_return: 81 (0.3%)

### Known data quirks

- **STI (SunTrust → Solidion):** Ticker reused after SunTrust merged into Truist (2019). Old SunTrust data ($539) and new Solidion data ($23) coexist. Extreme record deleted.
- **FRC y_return = -5.88:** First Republic Bank collapse ($122→$0.34). Real event, kept in data.
- **Partial y_return for latest quarter:** DB has y_return filled for 2025-12-31 using today's price as proxy. This is NOT a fully realized quarterly return and must NOT be used for training.

---

## Notes

- **Rebalance cadence:** Run with `--mixed-vintage` as earnings come in throughout the quarter. Early reporters get their latest data; others keep using previous quarter data. Re-run weekly during earnings season for fresher picks.
- **val-cutoff updates:** Advance `--val-cutoff` by one quarter when a new quarter's y_return becomes fully realized (i.e., both the current and next tradedates are in the past).
- **FMP API limits:** `limit=40` returns ~10 years of quarterly data. FMP Starter plan only returns recent 5 quarters regardless of `limit`.
- **Cache behavior:** `_fetch_fmp_data` uses a local-first strategy. If raw data exists in `data/cache/finrl_trading.db`, it skips the API call. Clear the cache to force a refresh.
- **y_return definition:** Forward log return `ln(P[t+1]/P[t])` where P is the quarter-end adjusted close. Positive = stock went up next quarter.
- **Ticker format:** FMP uses `-` (BRK-B), DB uses `.` (BRK.B). Conversion handled automatically.

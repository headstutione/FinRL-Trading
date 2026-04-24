# ML Stock Selection Pipeline

End-to-end guide for ML-driven stock selection with per-sector-bucket models.

**Database:** 22,909 records, 715 tickers, 64 columns (2015-Q2 ~ 2026-Q1)

**Output:** Ranked stock picks per bucket (growth_tech, cyclical, real_assets, defensive) with 7 competing ML models.

---

## CRITICAL SPEC 1: y_return Calculation

> **This is the single most important calculation in the entire ML strategy. Always verify y_return correctness before running any model.**

### Definition

```
y_return[i] = ln(next_trade_price / this_trade_price)
```

y_return is the **quarterly log return measured at tradedate prices** — the actual return an investor earns by buying at this tradedate and selling at the next tradedate.

### The Chain

```
datadate → tradedate → actual_tradedate → trade_price → y_return
```

| Step | Field | Meaning | Example (AMD Q3 2025) |
|------|-------|---------|----------------------|
| 1 | `datadate` | Quarter-end date of the financial report | 2025-09-30 |
| 2 | `tradedate` | Theoretical trade date (datadate + 2 months) | 2025-12-01 |
| 3 | `actual_tradedate` | First NYSE trading day on or after tradedate | 2025-12-01 |
| 4 | `trade_price` | adjClose on actual_tradedate | 219.76 |
| 5 | `y_return` | ln(next trade_price / this trade_price) | ln(198.62/219.76) = **-0.1011** |

### datadate → tradedate Mapping

| datadate (quarter end) | tradedate | Rationale |
|------------------------|-----------|----------|
| 03-31 | 06-01 (same year) | Q1 report available by June |
| 06-30 | 09-01 (same year) | Q2 report available by September |
| 09-30 | 12-01 (same year) | Q3 report available by December |
| 12-31 | 03-01 (next year) | Q4 report available by March |

### Worked Example (AMD)

```
datadate    tradedate    trade_price    y_return = ln(next/this)
─────────   ─────────    ───────────    ─────────────────────────
2025-03-31  2025-06-01   114.63         ln(162.32/114.63) = +0.3479
2025-06-30  2025-09-01   162.32         ln(219.76/162.32) = +0.3030
2025-09-30  2025-12-01   219.76         ln(198.62/219.76) = -0.1011
2025-12-31  2026-03-01   198.62         NULL (next tradedate not yet reached)
```

### Common Mistakes

| Wrong approach | Why it's wrong | Correct approach |
|----------------|----------------|------------------|
| `ln(adj_close_q[t+1] / adj_close_q[t])` | adj_close_q is the quarter-end price; the investor cannot buy at quarter-end because the report isn't public yet | Use `trade_price` (price at tradedate) |
| `ln(price_2025-12-31 / price_2025-09-30)` | On 2025-09-30 the Q3 report is not yet public; you can't act on it until tradedate 2025-12-01 | Buy at tradedate, not quarter-end |
| y_return = 0 | Frozen price from delisted ticker whose adj_close_q was never updated | Set to NULL |

> **Personal note:** I initially made the `adj_close_q` mistake on my first pass — the y_return values looked plausible but were systematically off because they ignored the 2-month reporting lag. The tradedate offset is non-negotiable; don't skip it even for quick experiments.

### Pre-Run Verification (MUST execute before every model run)

```python
import sqlite3, pandas as pd, numpy as np
conn = sqlite3.connect('data/finrl_trading.db')
df = pd.read_sql('''
    SELECT ticker, datadate, trade
```

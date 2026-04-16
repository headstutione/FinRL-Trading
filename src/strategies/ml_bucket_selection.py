#!/usr/bin/env python3
"""Per-bucket ML stock selection: train on all tickers <=2025, predict Q1 2026 (or latest quarter with data)."""

import argparse
import os
import sqlite3
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Allow running as standalone script: python3 src/strategies/ml_bucket_selection.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sector -> Bucket mapping (synced with group_selection_by_gics.py v1.2.2)
# ---------------------------------------------------------------------------
SECTOR_TO_BUCKET = {
    "information technology": "growth_tech",
    "technology": "growth_tech",
    "communication services": "growth_tech",
    "consumer discretionary": "cyclical",
    "consumer cyclical": "cyclical",
    "financials": "cyclical",
    "financial services": "cyclical",
    "industrials": "cyclical",
    "energy": "real_assets",
    "materials": "real_assets",
    "basic materials": "real_assets",
    "real estate": "real_assets",
    "health care": "defensive",
    "healthcare": "defensive",
    "consumer staples": "defensive",
    "consumer defensive": "defensive",
    "utilities": "defensive",
}

FEATURE_COLS = [
    # Valuation (5)
    "pe", "ps", "pb", "peg", "ev_multiple",
    # Profitability (4)
    "EPS", "roe", "gross_margin", "operating_margin",
    # Cash Flow (5)
    "fcf_per_share", "cash_per_share", "capex_per_share", "fcf_to_ocf", "ocf_ratio",
    # Leverage (3)
    "debt_ratio", "debt_to_equity", "debt_to_mktcap",
    # Liquidity (1)
    "cur_ratio",
    # Efficiency (3)
    "acc_rec_turnover", "asset_turnover", "payables_turnover",
    # Coverage (2)
    "interest_coverage", "debt_service_coverage",
    # Dividend (1)
    "dividend_yield",
    # Solvency (1)
    "solvency_ratio",
    # Per-Share (1)
    "BPS",
]

# Momentum features computed from sequential data (not stored in DB)
MOMENTUM_COLS = [
    # Price momentum (from adj_close_q)
    "ret_1q", "ret_2q", "ret_4q",
    # Fundamental momentum (QoQ changes)
    "eps_chg", "roe_chg", "gm_chg", "om_chg",
]


def build_models():
    models = {
        "RF": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "XGB": None,
        "LGBM": None,
        "HistGBM": HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
    }
    try:
        from xgboost import XGBRegressor
        models["XGB"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    except ImportError:
        del models["XGB"]

    try:
        from lightgbm import LGBMRegressor
        models["LGBM"] = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    except ImportError:
        del models["LGBM"]

    return models


def run_bucket(bucket, bdf, feature_cols, val_cutoff="2025-12-31", val_quarters=3):
    """Train models for one bucket, return (predictions_df, model_results_list)."""

    # Validation: last N quarters up to val_cutoff (inclusive)
    all_dates = sorted(bdf[bdf["y_return"].notna()]["datadate"].unique())
    val_end_idx = None
    for i, d in enumerate(all_dates):
        if str(d) <= val_cutoff:
            val_end_idx = i
    if val_end_idx is not None:
        val_start_idx = max(0, val_end_idx - val_quarters + 1)
        val_dates = set(all_dates[val_start_idx : val_end_idx + 1])
    else:
        val_dates = set()

    train_b = bdf[(~bdf["datadate"].isin(val_dates)) & (bdf["datadate"] <= val_cutoff) & (bdf["y_return"].notna())]
    val_b = bdf[(bdf["datadate"].isin(val_dates)) & (bdf["y_return"].notna())]
    # Infer on ALL quarters after val_cutoff
    infer_dates = sorted(bdf[bdf["datadate"] > val_cutoff]["datadate"].unique())
    if infer_dates:
        infer_b = bdf[bdf["datadate"].isin(infer_dates)]
    else:
        infer_b = pd.DataFrame()

    print(f"\n{'=' * 60}")
    print(f"  Bucket: {bucket.upper()}")
    val_date_range = f"{sorted(val_dates)[0]} ~ {sorted(val_dates)[-1]}" if val_dates else "none"
    print(f"  Train: {len(train_b)} | Val: {len(val_b)} ({len(val_dates)}Q: {val_date_range}) | Infer: {len(infer_b)}")
    if len(infer_b) > 0:
        print(f"  Infer dates: {infer_dates} ({len(infer_dates)}Q)")
    print(f"{'=' * 60}")

    if len(train_b) < 20 or len(infer_b) == 0:
        print("  SKIP: insufficient data")
        return pd.DataFrame(), [], []

    X_train, y_train = train_b[feature_cols].values, train_b["y_return"].values
    X_val, y_val = val_b[feature_cols].values, val_b["y_return"].values
    X_infer = infer_b[feature_cols].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(val_b) > 0 else None
    X_infer_s = scaler.transform(X_infer)

    models = build_models()
    fitted = {}
    model_results = []
    best_name, best_mse, best_model = None, float("inf"), None

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        fitted[name] = model
        if X_val_s is not None and len(X_val_s) > 0:
            mse = mean_squared_error(y_val, model.predict(X_val_s))
        else:
            mse = float("inf")
        model_results.append({
            "bucket": bucket, "model": name, "val_mse": round(mse, 6),
            "train_size": len(train_b), "val_size": len(val_b), "infer_size": len(infer_b),
        })
        print(f"  {name:12s}: MSE = {mse:.6f}")
        if mse < best_mse:
            best_name, best_mse, best_model = name, mse, model

    # Stacking top 3
    if X_val_s is not None and len(X_val_s) > 0:
        sorted_m = sorted(
            [(n, mean_squared_error(y_val, fitted[n].predict(X_val_s))) for n in fitted],
            key=lambda x: x[1],
        )
    else:
        sorted_m = [(n, 0) for n in fitted]
    top3 = [n for n, _ in sorted_m[:3]]
    stacking = StackingRegressor(
        estimators=[(n, fitted[n]) for n in top3],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1,
    )
    stacking.fit(X_train_s, y_train)
    fitted["Stacking"] = stacking
    if X_val_s is not None and len(X_val_s) > 0:
        stack_mse = mean_squared_error(y_val, stacking.predict(X_val_s))
    else:
        stack_mse = float("inf")
    model_results.append({
        "bucket": bucket, "model": "Stacking", "val_mse": round(stack_mse, 6),
        "train_size": len(train_b), "val_size": len(val_b), "infer_size": len(infer_b),
    })
    print(f"  {'Stacking':12s}: MSE = {stack_mse:.6f}  (base: {top3})")

    if stack_mse < best_mse:
        best_name, best_mse, best_model = "Stacking", stack_mse, stacking

    print(f"  >> Best: {best_name} (MSE={best_mse:.6f})")

    # Retrain all models on train + val before inference
    full_train = pd.concat([train_b, val_b], ignore_index=True)
    X_full, y_full = full_train[feature_cols].values, full_train["y_return"].values
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_full)
    X_infer_s = scaler_full.transform(X_infer)
    print(f"  Retrained on train+val: {len(full_train)} samples")

    for name, model in fitted.items():
        if name == "Stacking":
            continue  # rebuild stacking below
        model.fit(X_full_s, y_full)
    # Rebuild stacking with retrained base models
    stacking = StackingRegressor(
        estimators=[(n, fitted[n]) for n in top3],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1,
    )
    stacking.fit(X_full_s, y_full)
    fitted["Stacking"] = stacking
    if best_name == "Stacking":
        best_model = stacking

    # Predict
    infer_b = infer_b.copy()
    infer_b["predicted_return"] = best_model.predict(X_infer_s)
    infer_b["best_model"] = best_name
    for n, m in fitted.items():
        infer_b[f"pred_{n}"] = m.predict(X_infer_s)

    # Inverse-MSE weighted ensemble (weights from val MSE, predictions from retrained models)
    mse_map = {r["model"]: r["val_mse"] for r in model_results}
    pred_model_cols = [c for c in infer_b.columns if c.startswith("pred_") and c != "pred_ensemble_avg"]
    weights = {}
    for col in pred_model_cols:
        name = col.replace("pred_", "")
        mse = mse_map.get(name, None)
        weights[col] = (1.0 / mse) if mse and mse > 0 else 0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    infer_b["pred_ensemble_avg"] = sum(infer_b[col] * w for col, w in weights.items())

    infer_b = infer_b.sort_values(["datadate", "predicted_return"], ascending=[True, False])

    # Print ranking per quarter
    for idate in infer_dates:
        qdf = infer_b[infer_b["datadate"] == idate].sort_values("predicted_return", ascending=False)
        actual_col = "y_return" if "y_return" in qdf.columns and qdf["y_return"].notna().any() else None
        print(f"\n  Ranking ({idate}, {len(qdf)} stocks):")
        for i, (_, r) in enumerate(qdf.head(10).iterrows()):
            marker = " ***" if i < 3 else ""
            actual = f"  actual={r['y_return'] * 100:+.1f}%" if actual_col and pd.notna(r.get("y_return")) else ""
            print(f"    {i + 1:2d}. {r['tic']:6s}  pred={r['predicted_return'] * 100:+6.1f}%{actual}{marker}")

    # Feature importance (collect from all models that expose it)
    importance_records = []
    for name, model in fitted.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            for rank_idx, (feat, val) in enumerate(imp.items(), 1):
                importance_records.append({
                    "bucket": bucket, "model": name,
                    "is_best": name == best_name,
                    "feature": feat, "importance": round(val, 6),
                    "rank": rank_idx,
                })
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            imp = pd.Series(coefs, index=feature_cols).sort_values(ascending=False)
            total = imp.sum()
            for rank_idx, (feat, val) in enumerate(imp.items(), 1):
                importance_records.append({
                    "bucket": bucket, "model": name,
                    "is_best": name == best_name,
                    "feature": feat, "importance": round(val / total if total > 0 else 0, 6),
                    "rank": rank_idx,
                })

    # Print top 5 for best model
    best_imp = [r for r in importance_records if r["model"] == best_name]
    if best_imp:
        best_imp_sorted = sorted(best_imp, key=lambda x: x["importance"], reverse=True)
        print(f"\n  Top 5 Features ({best_name}):")
        for r in best_imp_sorted[:5]:
            print(f"    {r['feature']:20s} {r['importance']:.3f}")

    return infer_b, model_results, importance_records


def main():
    parser = argparse.ArgumentParser(description="Per-bucket ML stock selection")
    parser.add_argument("--db", default=os.path.join(project_root, "data", "finrl_trading.db"))
    parser.add_argument("--universe", default=None,
                        help="Filter to a stock universe: sp500, nasdaq100, or path to CSV with 'tickers' column")
    parser.add_argument("--val-cutoff", default="2025-12-31", help="Validation end date (last val quarter)")
    parser.add_argument("--val-quarters", type=int, default=3, help="Number of validation quarters (default: 3)")
    parser.add_argument("--output-dir", default=os.path.join(project_root, "data"))
    parser.add_argument("--survivorship-free", action="store_true",
                        help="Filter training data by historical SP500 membership per quarter (no filter on inference)")
    parser.add_argument("--latest-snapshot", action="store_true",
                        help="Inference using latest available filing per ticker (fill missing with previous quarter)")
    parser.add_argument("--mixed-vintage", action="store_true",
                        help="Inference: use latest available datadate per current-SP500 ticker (mix Q4/Q1)")
    parser.add_argument("--ref-date", default=None,
                        help="Reference date for actual return calculation in latest-snapshot mode (default: last quarter end)")
    parser.add_argument("--end-date", default=None,
                        help="End date for return calculation (default: today)")
    args = parser.parse_args()

    # Load data
    conn = sqlite3.connect(args.db)
    _feat_sql = ", ".join(FEATURE_COLS)
    df = pd.read_sql(
        f"""SELECT ticker as tic, datadate, gsector, adj_close_q,
           filing_date, accepted_date,
           {_feat_sql}, y_return
           FROM fundamental_data ORDER BY ticker, datadate""",
        conn,
    )
    conn.close()

    # Filter to universe if specified
    if args.universe:
        if args.universe.lower() == "nasdaq100":
            import sys as _sys; _sys.path.insert(0, os.path.join(project_root, "src"))
            from data.data_fetcher import fetch_nasdaq100_tickers
            univ = fetch_nasdaq100_tickers()
            univ_tickers = set(univ["tickers"].tolist())
        elif args.universe.lower() == "sp500":
            from data.data_fetcher import fetch_sp500_tickers
            univ = fetch_sp500_tickers()
            univ_tickers = set(univ["tickers"].tolist())
        elif os.path.exists(args.universe):
            univ_tickers = set(pd.read_csv(args.universe)["tickers"].tolist())
        else:
            raise ValueError(f"Unknown universe: {args.universe}")
        before = len(df)
        df = df[df["tic"].isin(univ_tickers)].copy()
        print(f"Universe filter ({args.universe}): {before} -> {len(df)} records ({df['tic'].nunique()} tickers)")

    print(f"Loaded {len(df)} records, {df['tic'].nunique()} tickers")
    print(f"Date range: {df['datadate'].min()} ~ {df['datadate'].max()}")
    print(f"Val cutoff: {args.val_cutoff}")

    # Survivorship-free filtering: for training/val quarters, keep only tickers
    # that were actually in SP500 at that time. Inference quarters are NOT filtered.
    if args.survivorship_free:
        from data.data_fetcher import get_sp500_members_at_date
        hist_csv = os.path.join(project_root, "data", "sp500_historical_constituents.csv")
        hist_df = pd.read_csv(hist_csv)
        hist_df['date'] = pd.to_datetime(hist_df['date'])

        all_quarters = sorted(df["datadate"].unique())
        membership = {}
        for q in all_quarters:
            q_dt = pd.to_datetime(q)
            valid = hist_df[hist_df['date'] <= q_dt]
            if not valid.empty:
                membership[q] = set(valid.iloc[-1]['tickers'].split(','))
            else:
                membership[q] = set()

        # Mark rows: True if ticker was in SP500 at that quarter, or if it's an inference quarter
        before = len(df)
        is_infer = df["datadate"] > args.val_cutoff
        is_member = df.apply(lambda r: r['tic'] in membership.get(r['datadate'], set()), axis=1)
        df = df[is_infer | is_member].copy()
        removed = before - len(df)
        print(f"Survivorship-free filter: {before} -> {len(df)} records (removed {removed} non-member rows)")

    # Momentum features: price momentum + fundamental QoQ changes
    df = df.sort_values(["tic", "datadate"]).copy()
    df["adj_close_q"] = pd.to_numeric(df["adj_close_q"], errors="coerce")
    df["ret_1q"] = df.groupby("tic")["adj_close_q"].pct_change(1)
    df["ret_2q"] = df.groupby("tic")["adj_close_q"].pct_change(2)
    df["ret_4q"] = df.groupby("tic")["adj_close_q"].pct_change(4)
    for src, dst in [("EPS", "eps_chg"), ("roe", "roe_chg"),
                     ("gross_margin", "gm_chg"), ("operating_margin", "om_chg")]:
        df[src] = pd.to_numeric(df[src], errors="coerce")
        df[dst] = df.groupby("tic")[src].diff()
    print(f"Added {len(MOMENTUM_COLS)} momentum features: {MOMENTUM_COLS}")

    # Prep features
    all_feat = FEATURE_COLS + MOMENTUM_COLS
    for c in all_feat:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    global_medians = df[all_feat].median()
    df[all_feat] = df[all_feat].fillna(global_medians).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Winsorize: clip at 1st/99th percentile to reduce outlier impact
    for c in all_feat:
        p01, p99 = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lower=p01, upper=p99)

    # Assign buckets
    df["bucket"] = df["gsector"].str.lower().map(SECTOR_TO_BUCKET)
    unmapped = df[df["bucket"].isna()]["gsector"].unique()
    if len(unmapped) > 0:
        print(f"WARNING: unmapped sectors: {unmapped}")
    df = df[df["bucket"].notna()].copy()

    # Latest-snapshot mode: for inference, use the most recent filing per ticker
    # instead of only future quarter dates. This gives ~500 stocks to rank.
    if args.latest_snapshot:
        from data.data_fetcher import fetch_sp500_tickers
        sp500 = fetch_sp500_tickers()
        sp500_tickers = set(sp500["tickers"].tolist()) if sp500 is not None else set()
        print(f"\nLatest-snapshot mode: {len(sp500_tickers)} current SP500 tickers")

        # For each SP500 ticker, take the latest available record (up to ref_date if set)
        import yfinance as yf
        from datetime import date
        ref_date = pd.Timestamp(args.ref_date) if args.ref_date else pd.Timestamp("2026-03-31")
        end_date = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp(date.today())

        train_part = df[df["datadate"] <= args.val_cutoff].copy()
        latest_rows = []
        for tic in sorted(sp500_tickers):
            tic_df = df[(df["tic"] == tic) & (df["datadate"] <= ref_date.strftime("%Y-%m-%d"))]
            if len(tic_df) == 0:
                continue
            latest = tic_df.sort_values("datadate").iloc[-1].copy()
            latest["datadate"] = "latest"  # synthetic date for inference
            latest_rows.append(latest)

        if latest_rows:
            snapshot_df = pd.DataFrame(latest_rows)
            # Download actual returns: price on ref_date -> end_date via FMP
            import requests
            from data.data_fetcher import FMPFetcher
            fmp = FMPFetcher()
            all_tickers_list = list(snapshot_df["tic"].unique())
            dl_start = (ref_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            dl_end = (end_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            print(f"  Ref: {ref_date.strftime('%Y-%m-%d')} -> End: {end_date.strftime('%Y-%m-%d')}")
            print(f"  Downloading prices via FMP for {len(all_tickers_list)} tickers ...")

            # Use FMP historical-price-eod API per ticker
            all_close = {}  # tic -> pd.Series(date->close)
            for i, tic in enumerate(all_tickers_list):
                try:
                    url = f"{fmp.base_url}/historical-price-eod/full?symbol={tic}&from={dl_start}&to={dl_end}&apikey={fmp.api_key}"
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    rows = data.get("historical", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    if rows:
                        s = pd.Series(
                            {pd.Timestamp(r["date"]): r["adjClose"] if "adjClose" in r else r["close"] for r in rows}
                        ).sort_index()
                        if len(s) > 0:
                            all_close[tic] = s
                except Exception:
                    pass
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(all_tickers_list)} done ...")

            print(f"  Got prices for {len(all_close)} / {len(all_tickers_list)} tickers")

            # Get price at ref_date and end_date for each ticker
            # Use first trading day ON or AFTER the date (consistent with DB tradedate logic)
            y_returns = {}
            for tic in snapshot_df["tic"].unique():
                if tic not in all_close:
                    continue
                series = all_close[tic]
                # Buy: first trading day >= ref_date
                mask_ref = series.index >= ref_date
                if not mask_ref.any():
                    continue
                p_ref = float(series[mask_ref].iloc[0])
                # Sell: first trading day >= end_date; if future, use latest available
                mask_end = series.index >= end_date
                if mask_end.any():
                    p_end = float(series[mask_end].iloc[0])
                else:
                    p_end = float(series.iloc[-1])  # end_date in future, use latest
                if p_ref > 0:
                    y_returns[tic] = np.log(p_end / p_ref)

            snapshot_df["y_return"] = snapshot_df["tic"].map(y_returns)
            filled = snapshot_df["y_return"].notna().sum()
            print(f"  Snapshot: {len(snapshot_df)} tickers, y_return filled: {filled}")

            # Replace inference data: keep train_part + snapshot
            df = pd.concat([train_part, snapshot_df], ignore_index=True)

    # Mixed-vintage mode: for inference, keep only the latest record per ticker
    # among current SP500 members. Early reporters get Q1 2026 data, rest keep Q4 2025.
    if args.mixed_vintage:
        hist_csv = os.path.join(project_root, "data", "sp500_historical_constituents.csv")
        _hist_mv = pd.read_csv(hist_csv)
        _hist_mv['date'] = pd.to_datetime(_hist_mv['date'])
        latest_sp500 = set(
            t.strip() for t in _hist_mv.loc[_hist_mv['date'] == _hist_mv['date'].max(), 'tickers'].iloc[0].split(',')
        )

        train_val_part = df[df["datadate"] <= args.val_cutoff].copy()
        infer_part = df[(df["datadate"] > args.val_cutoff) & (df["tic"].isin(latest_sp500))].copy()

        # Keep only latest datadate per ticker
        infer_part = infer_part.sort_values(["tic", "datadate"]).drop_duplicates(subset="tic", keep="last")

        # Tag vintage for output
        infer_part["data_vintage"] = infer_part["datadate"].apply(
            lambda d: "Q1_2026" if d >= "2026-03-01" else "Q4_2025"
        )

        n_q1 = (infer_part["data_vintage"] == "Q1_2026").sum()
        n_q4 = (infer_part["data_vintage"] == "Q4_2025").sum()
        print(f"\nMixed-vintage mode: {len(infer_part)} current SP500 tickers for inference")
        print(f"  Q1 2026 (early reporters): {n_q1}")
        print(f"  Q4 2025 (not yet reported): {n_q4}")

        # Set all inference records to a common synthetic datadate so they rank together
        infer_part["original_datadate"] = infer_part["datadate"]
        infer_part["datadate"] = "mixed"

        df = pd.concat([train_val_part, infer_part], ignore_index=True)

    # Sub-sector indicator features: one-hot encode gsector so models can
    # distinguish sectors within each bucket (e.g. Energy vs Real Estate).
    sector_dummies = pd.get_dummies(df["gsector"], prefix="sector")
    df = pd.concat([df, sector_dummies], axis=1)

    # Run per bucket
    all_preds = []
    all_model_results = []
    all_importances = []

    for bucket in ["growth_tech", "cyclical", "real_assets", "defensive"]:
        bdf = df[df["bucket"] == bucket].copy()
        # Keep only sector dummy columns that have variance within this bucket
        sector_cols = [c for c in bdf.columns if c.startswith("sector_") and bdf[c].sum() > 0]
        mom_cols = MOMENTUM_COLS if bucket == "growth_tech" else []
        bucket_features = FEATURE_COLS + mom_cols + sector_cols
        print(f"\n  [Features for {bucket}]: {len(FEATURE_COLS)} fundamental + {len(mom_cols)} momentum + {len(sector_cols)} sector = {len(bucket_features)}")
        preds, results, importances = run_bucket(bucket, bdf, bucket_features, val_cutoff=args.val_cutoff, val_quarters=args.val_quarters)
        if len(preds) > 0:
            all_preds.append(preds)
        all_model_results.extend(results)
        all_importances.extend(importances)

    if not all_preds:
        print("\nNo predictions generated.")
        return

    pred_all = pd.concat(all_preds, ignore_index=True)

    # Per-bucket-per-quarter ranking
    pred_all["rank_best"] = pred_all.groupby(["bucket", "datadate"])["predicted_return"].rank(ascending=False).astype(int)
    pred_all["rank_ensemble"] = pred_all.groupby(["bucket", "datadate"])["pred_ensemble_avg"].rank(ascending=False).astype(int)

    # Save — prefix filenames with universe name + timestamp
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{args.universe}_" if args.universe else "sp500_"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    pred_path = os.path.join(args.output_dir, f"{prefix}ml_bucket_predictions_{timestamp}.csv")
    pred_all.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path} ({len(pred_all)} stocks)")

    model_path = os.path.join(args.output_dir, f"{prefix}ml_bucket_model_results_{timestamp}.csv")
    pd.DataFrame(all_model_results).to_csv(model_path, index=False)

    if all_importances:
        imp_df = pd.DataFrame(all_importances)
        imp_df = imp_df.sort_values(["bucket", "model", "rank"])
        imp_path = os.path.join(args.output_dir, f"{prefix}ml_feature_importance_{timestamp}.csv")
        imp_df.to_csv(imp_path, index=False)
        print(f"Saved: {imp_path} ({len(imp_df)} rows)")
    print(f"Saved: {model_path} ({len(all_model_results)} rows)")

    # Summary per quarter
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: Top picks per bucket per quarter")
    print(f"{'=' * 60}")
    for idate in sorted(pred_all["datadate"].unique()):
        print(f"\n  --- {idate} ---")
        for bucket in ["growth_tech", "cyclical", "real_assets", "defensive"]:
            bp = pred_all[(pred_all["bucket"] == bucket) & (pred_all["datadate"] == idate)].sort_values("rank_best").head(3)
            if len(bp) == 0:
                print(f"  {bucket:15s}: no data")
                continue
            best_m = bp.iloc[0]["best_model"]
            picks = ", ".join(
                f"{r['tic']}({r['predicted_return'] * 100:+.1f}%)" for _, r in bp.iterrows()
            )
            print(f"  {bucket:15s} [{best_m}]: {picks}")


if __name__ == "__main__":
    main()

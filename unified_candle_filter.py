
# -*- coding: utf-8 -*-
"""
Unified Candle Filter (Drop-in)
--------------------------------
Ersetzt die drei Filter:
  - remove_micro_flip_candles
  - remove_tiny_same_color_candles
  - remove_isolated_candles

Prinzip (simpel & tuning-freundlich):
  Entferne Kerzen, deren Range (High-Low) UND Body (|Close-Open|) relativ
  zur rollenden Median-Range des Umfelds klein sind.

Integration:
  - Diesen Block in trend_calculation.py einfügen (oder importieren).
  - Aufruf ersetzt die alte Dreierkette an EINER Stelle:
      df, report = unify_candle_filters(df, W=50, tau_range=0.12, tau_body=0.08,
                                        pivot_protect=True, pivot_look=2,
                                        reindex_sequential=False, return_report=True)

Rückgabe:
  - df_gefiltert (DataFrame), unveränderte Spalten
  - report (dict) mit Kennzahlen (optional)
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

def unify_candle_filters(
    df: pd.DataFrame,
    *,
    W: int = 50,                 # Fensterbreite für rollende Median-Range
    tau_range: float = 0.12,     # Range-Schwelle relativ zur Baseline
    tau_body: float = 0.08,      # Body-Schwelle relativ zur Baseline
    pivot_protect: bool = True,  # lokale High/Low-Pivots nie entfernen
    pivot_look: int = 2,         # Nachbarschaftsgröße für Pivots (links/rechts)
    reindex_sequential: bool = False,  # Kerze_Nr/Index neu 0..N-1 setzen
    return_report: bool = True   # Report zurückgeben
) -> Tuple[pd.DataFrame, Dict[str, Any]] | pd.DataFrame:
    """
    Entfernt „kleine“ Kerzen relativ zum Umfeld.
    Bedingungen (AND):
        Range <= tau_range * Baseline
        Body  <= tau_body  * Baseline
    Baseline: rollende Median-Range über Fenster W (robust).
    """
    if df is None or len(df) == 0:
        return (df.copy(), {"initial_len": 0, "final_len": 0, "removed_count": 0}) if return_report else df.copy()

    work = df.copy()

    # --- Spalten prüfen & numerisch machen ---
    needed = ["Open", "High", "Low", "Close"]
    missing = [c for c in needed if c not in work.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten für unify_candle_filters: {missing}")

    for c in needed:
        if not pd.api.types.is_float_dtype(work[c]):
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # --- Kenngrößen ---
    rng = (work["High"] - work["Low"]).astype(float)
    body = (work["Close"] - work["Open"]).abs().astype(float)

    # --- Baseline: rollende Median-Range ---
    minp = max(5, W // 3)
    base = rng.rolling(window=W, center=True, min_periods=minp).median()

    # Fallback, wenn zu viele NaNs/Nullen
    global_med = float(np.nanmedian(rng)) if len(rng) else 0.0
    if not np.isfinite(global_med) or global_med <= 0.0:
        # minimaler Epsilon aus Kursgröße ableiten
        close_med = float(np.nanmedian(work["Close"])) if len(work["Close"]) else 1.0
        global_med = max(close_med * 1e-6, 1e-9)

    base = base.fillna(global_med)
    base = base.where(base > 0.0, other=global_med)

    # --- „klein“-Maske ---
    m1 = rng <= (tau_range * base)
    m2 = body <= (tau_body * base)
    remove_mask = (m1 & m2)

    # --- Pivots schützen (lokale Extrema) ---
    if pivot_protect and len(work) >= (2 * pivot_look + 1):
        # Rolling Max/Min mit centered Fenster; True, wenn Kerze lokales Extrem ist
        win = 2 * pivot_look + 1
        roll_max = work["High"].rolling(window=win, center=True, min_periods=win).max()
        roll_min = work["Low"].rolling(window=win, center=True, min_periods=win).min()
        is_pivot_high = (work["High"] >= roll_max) & roll_max.notna()
        is_pivot_low  = (work["Low"]  <= roll_min) & roll_min.notna()
        pivot_mask = (is_pivot_high | is_pivot_low).fillna(False)
        remove_mask = remove_mask & (~pivot_mask)

    keep_mask = ~remove_mask

    result = work.loc[keep_mask].copy()

    # Optional: Reindex/Kerze_Nr neu setzen (für indexbasierten Plot)
    if reindex_sequential:
        result = result.reset_index(drop=True)
        if "Kerze_Nr" in result.columns:
            result["Kerze_Nr"] = np.arange(len(result), dtype=int)

    if return_report:
        report = {
            "initial_len": int(len(work)),
            "final_len": int(len(result)),
            "removed_count": int(remove_mask.sum(skipna=True)),
            "params": {
                "W": W,
                "tau_range": float(tau_range),
                "tau_body": float(tau_body),
                "pivot_protect": bool(pivot_protect),
                "pivot_look": int(pivot_look),
                "reindex_sequential": bool(reindex_sequential),
            }
        }
        return result, report

    return result

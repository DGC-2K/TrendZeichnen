"""
trend_detection.py
------------------
Trend-Erkennung gemäß Vorgabe:
- Aufwärtstrend:  A < C < B < E < D < F
- Abwärtstrend:   A > C > B > E > D > F
- sonst:          Kein klarer Trend

Integration in bestehende Architektur:
- nutzt ArmContainer/ArmConnection und berechne_verbindungslinien aus trend_calculation
- extrahiert die C-Pivotfolge (A,B,C,D,...) aus den berechneten Verbindungslinien
- liefert eine gleitende Bewertung über jeweils 6 aufeinanderfolgende Pivotpreise
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from trend_calculation import (
    ArmContainer,
    ArmConnection,
    berechne_verbindungslinien,
)

BASE_OUTPUT_DIR = r"D:\TradingBot\output"


# --- Kernfunktion nach deinem Pseudocode -------------------------------------------------

def erkenneTrend(pA: float, pB: float, pC: float, pD: float, pE: float, pF: float) -> str:
    """
    Gibt 'Aufwärtstrend', 'Abwärtstrend' oder 'Kein klarer Trend' zurück.
    Regeln (exakt wie vorgegeben):
        UP:   A < C < B < E < D < F
        DOWN: A > C > B > E > D > F
    """
    try:
        if (
            (pA < pC) and (pC < pB) and
            (pB < pE) and (pE < pD) and
            (pD < pF)
        ):
            return "Aufwärtstrend"
        elif (
            (pA > pC) and (pC > pB) and
            (pB > pE) and (pE > pD) and
            (pD > pF)
        ):
            return "Abwärtstrend"
    except TypeError:
        # Falls einer None/NaN ist etc.
        pass
    return "Kein klarer Trend"


# --- Hilfen zum Ermitteln der Pivotfolge aus den Verbindungslinien ----------------------

def _build_pivots_from_verbindungen(verbindungen_liste: List[dict]) -> List[Tuple[int, float]]:
    """
    Erzeugt eine geordnete Liste von (pivot_idx, pivot_price) aus der
    verbindungen_liste (wie sie berechne_verbindungslinien liefert).

    Regeln:
    - Bei 'B1' und 'B-C': (start) -> (ende)
    - Bei 'B-D-C': (start) -> (mitte) -> (ende)
    - Doppelte Startpunkte werden vermieden (z. B. wenn der nächste Eintrag dort wieder ansetzt).
    """
    pivots: List[Tuple[int, float]] = []

    def _append(tp: Tuple[int, float]) -> None:
        idx, price = tp
        if price is None or np.isnan(float(price)):
            return
        if len(pivots) == 0 or pivots[-1][0] != idx:
            pivots.append((int(idx), float(price)))

    for v in verbindungen_liste:
        typ = v.get("typ")
        if typ in ("B1", "B-C"):
            start = v.get("start")  # (idx, price)
            ende  = v.get("ende")
            if start is not None and ende is not None:
                _append(start)
                _append(ende)
        elif typ == "B-D-C":
            start = v.get("start")
            mitte = v.get("mitte")
            ende  = v.get("ende")
            if start is not None and mitte is not None and ende is not None:
                _append(start)
                _append(mitte)
                _append(ende)
        else:
            # unbekannter Typ – sicherheitshalber Start/Ende, falls vorhanden
            if v.get("start"):
                _append(v["start"])
            if v.get("ende"):
                _append(v["ende"])

    return pivots


def extract_c_pivots(ha_data: pd.DataFrame, arm_container: ArmContainer) -> List[Tuple[int, float]]:
    """
    Leitet aus den validierten Armen des ArmContainer die Verbindungsserie ab
    und konvertiert sie in eine sequenzielle Pivotliste:
    [(A_idx, A_price), (B_idx, B_price), (C_idx, C_price), ...]
    """
    validated_arms = [a for a in getattr(arm_container, "arms", []) if getattr(a, "validated", False)]
    if not validated_arms:
        validated_arms = list(getattr(arm_container, "arms", []))

    verbindungen_liste = berechne_verbindungslinien(validated_arms, ha_data)
    return _build_pivots_from_verbindungen(verbindungen_liste)


# --- Öffentliche API: Trend-Erkennung über gleitende 6er-Pivotfenster -------------------

def detect_trends_from_armcontainer(
    ha_data: pd.DataFrame,
    arm_container: ArmContainer,
    only_validated: bool = True,
) -> List[Dict]:
    """
    Erzeugt aus ArmContainer/ha_data die C-Pivotsequenz und bewertet jedes
    gleitende Fenster aus 6 Pivotpreisen mit erkenneTrend().

    Returns
    -------
    Liste von Dicts mit:
      {
        "window_index": i,
        "A_idx": int, "A": float,
        "B_idx": int, "B": float,
        "C_idx": int, "C": float,
        "D_idx": int, "D": float,
        "E_idx": int, "E": float,
        "F_idx": int, "F": float,
        "trend": "Aufwärtstrend" | "Abwärtstrend" | "Kein klarer Trend"
      }
    """
    arms = [a for a in getattr(arm_container, "arms", []) if (a.validated if only_validated else True)]
    if not arms:
        return []

    # Verbindungslinien und daraus die Pivotfolge bilden
    verbindungen_liste = berechne_verbindungslinien(arms, ha_data)
    pivots = _build_pivots_from_verbindungen(verbindungen_liste)

    results: List[Dict] = []
    if len(pivots) < 6:
        return results

    # Gleitendes Fenster
    for i in range(len(pivots) - 5):
        (A_idx, A) = pivots[i + 0]
        (B_idx, B) = pivots[i + 1]
        (C_idx, C) = pivots[i + 2]
        (D_idx, D) = pivots[i + 3]
        (E_idx, E) = pivots[i + 4]
        (F_idx, F) = pivots[i + 5]

        trend = erkenneTrend(A, B, C, D, E, F)

        results.append({
            "window_index": i,
            "A_idx": A_idx, "A": A,
            "B_idx": B_idx, "B": B,
            "C_idx": C_idx, "C": C,
            "D_idx": D_idx, "D": D,
            "E_idx": E_idx, "E": E,
            "F_idx": F_idx, "F": F,
            "trend": trend,
        })

    return results


# --- Optional: Report als TXT/CSV -------------------------------------------------------

def write_trend_report(
    results: List[Dict],
    output_dir: str = BASE_OUTPUT_DIR,
    ticker: Optional[str] = None,
    prefix: str = "TrendErkennung"
) -> Tuple[str, str]:
    """
    Schreibt die Ergebnisse in CSV und eine lesbare TXT-Zusammenfassung.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{ticker}" if ticker else ""

    csv_path = os.path.join(output_dir, f"{prefix}{suffix}_{ts}.csv")
    txt_path = os.path.join(output_dir, f"{prefix}{suffix}_{ts}.txt")

    # CSV
    df = pd.DataFrame(results)
    if df.empty:
        df = pd.DataFrame(columns=["window_index","A_idx","A","B_idx","B","C_idx","C","D_idx","D","E_idx","E","F_idx","F","trend"])
    df.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig", float_format="%.5f")

    # TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"== Trend-Erkennung Bericht ==\n")
        f.write(f"Anzahl Fenster: {len(results)}\n\n")
        for r in results:
            f.write(
                f"[{r['window_index']:03d}] "
                f"A({r['A_idx']})={r['A']:.5f}  "
                f"B({r['B_idx']})={r['B']:.5f}  "
                f"C({r['C_idx']})={r['C']:.5f}  "
                f"D({r['D_idx']})={r['D']:.5f}  "
                f"E({r['E_idx']})={r['E']:.5f}  "
                f"F({r['F_idx']})={r['F']:.5f}  ->  {r['trend']}\n"
            )

    return csv_path, txt_path


# --- Beispiel (auskommentiert) ----------------------------------------------------------
# def example_usage(ha_data: pd.DataFrame, arm_container: ArmContainer):
#     results = detect_trends_from_armcontainer(ha_data, arm_container, only_validated=True)
#     csv_path, txt_path = write_trend_report(results, output_dir=BASE_OUTPUT_DIR, ticker="EURUSD=X")
#     print("Berichte:", csv_path, txt_path)

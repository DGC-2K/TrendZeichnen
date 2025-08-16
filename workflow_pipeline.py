import os
import pandas as pd
from typing import Tuple, List
from trend_calculation import calculate_ha, detect_trend_arms
from unified_candle_filter import unify_candle_filters
import json

def workflow_pipeline(rohdaten: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    """
    FÃ¼hrt den gesamten Workflow aus, schreibt Debug-CSV-Ausgaben und gibt
    die berechneten Heikin-Ashi-Daten und die Trendarme zurÃ¼ck.
    """

    # Basispfad fÃ¼r Debug/CSV-Dateien
    base_csv = r"D:\TradingBot\output\HA_Kerzen_Kontroll.csv"
    base, ext = os.path.splitext(base_csv)
    os.makedirs(os.path.dirname(base_csv), exist_ok=True)

    # 1. Rohdaten speichern
    rohdaten.head(50).to_csv(f"{base}_vor_calculate_ha{ext}", index=False)

    # 2. Heikin-Ashi-Berechnung
    ha_data = calculate_ha(rohdaten)
    ha_data.head(50).to_csv(f"{base}_nach_calculate_ha{ext}", index=False)

    # 3. Vor remove_isolated_candles
    ha_before = ha_data.copy()
    ha_before.head(50).to_csv(f"{base}_vor_remove_isolated_candles{ext}", index=False)

    # 4. remove_isolated_candles
    ha_data, report = unify_candle_filters(
        ha_data,
        W=50, tau_range=0.12, tau_body=0.08,
        pivot_protect=True, pivot_look=2,
        reindex_sequential=False, return_report=True
    )

# Nach dem Filter
ha_data.head(50).to_csv(f"{base}_nach_unified_filter{ext}", index=False)
    # 5. Entfernte Zeilen speichern (max 50)
    removed = pd.concat([ha_before, ha_data]).drop_duplicates(keep=False)
    removed.head(50).to_csv(f"{base}_entfernte_zeilen{ext}", index=False)

    # 6. Trendarme erkennen
    arms = detect_trend_arms(ha_data)

    # 7. Arms-Info als Textdatei speichern
    with open(f"{base}_trendarme.txt", "w", encoding="utf-8") as f:
        f.write(f"Anzahl gefundene Trendarme: {len(arms)}\n")
        for i, arm in enumerate(arms[:10]):  # Nur die ersten 10 ausgeben
            f.write(str(arm) + "\n")

    return ha_data, arms

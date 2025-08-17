import os
import pandas as pd
from typing import Tuple, List
from trend_calculation import calculate_ha, detect_trend_arms, remove_isolated_candles

def save_kontroll_csv(df: pd.DataFrame, path: str):
    def german_format(x):
        try:
            return f"{float(x):,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (ValueError, TypeError):
            if x is not None and str(x).strip() != '':
                return "0,000"
            return ''
    
    if 'Zeit' not in df.columns:
        raise ValueError("Die Spalte 'Zeit' muss im DataFrame vorhanden sein!")
    
    out = pd.DataFrame({
        'Kerze_Nr': df.index.values,
        'Zeit': pd.to_datetime(df['Zeit']).dt.strftime('%d.%m.%Y %H:%M'),
        'Open': df['Open'].apply(german_format),
        'High': df['High'].apply(german_format),
        'Low': df['Low'].apply(german_format),
        'Close': df['Close'].apply(german_format),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False, sep=";", encoding="utf-8-sig")

def workflow_pipeline(rohdaten: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    """
    Führt den gesamten Workflow aus, schreibt Debug-CSV-Ausgaben und gibt
    die berechneten Heikin-Ashi-Daten und die Trendarme zurück.
    """

    # Basispfad für Debug/CSV-Dateien
    base_csv = r"D:\TradingBot\output\HA_Kerzen_Kontroll.csv"
    base, ext = os.path.splitext(base_csv)
    os.makedirs(os.path.dirname(base_csv), exist_ok=True)

    # 1. Rohdaten speichern
    save_kontroll_csv(rohdaten.head(50), f"{base}_vor_calculate_ha{ext}")

    # 2. Heikin-Ashi-Berechnung
    ha_data = calculate_ha(rohdaten)
    save_kontroll_csv(ha_data.head(50), f"{base}_nach_calculate_ha{ext}")

    # 3. Vor remove_isolated_candles
    ha_before = ha_data.copy()
    save_kontroll_csv(ha_before.head(50), f"{base}_vor_remove_isolated_candles{ext}")

    # 4. remove_isolated_candles ausführen
    ha_data = remove_isolated_candles(ha_data)
    save_kontroll_csv(ha_data.head(50), f"{base}_nach_remove_isolated_candles{ext}")

    # 5. Entfernte Zeilen speichern (max 50)
    removed = pd.concat([ha_before, ha_data]).drop_duplicates(keep=False)
    save_kontroll_csv(removed.head(50), f"{base}_entfernte_zeilen{ext}")

    # 6. Trendarme erkennen
    arms = detect_trend_arms(ha_data)

    # 7. Arms-Info als Textdatei speichern
    with open(f"{base}_trendarme.txt", "w", encoding="utf-8") as f:
        f.write(f"Anzahl gefundene Trendarme: {len(arms)}\n")
        for i, arm in enumerate(arms[:10]):  # Nur die ersten 10 ausgeben
            f.write(str(arm) + "\n")

    return ha_data, arms
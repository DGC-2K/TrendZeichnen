#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- STANDARD LIBRARY IMPORTS ---
import os
from datetime import datetime, timedelta

# --- THIRD-PARTY LIBRARY IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt

# --- LOCAL MODULE IMPORTS ---
# Data processing
from data_processing import (
    download_data,
    prepare_data,
    bestimme_heikin_ashi_farbe,
)

# Trend calculation
from trend_calculation import (
    # HA & Guards
    calculate_ha,
    canonicalize_to_ha,
    remove_isolated_candles,
    remove_micro_flip_candles,
    count_isolated_ha_candles,
    # Arme & Verbindungen
    detect_trend_arms,
    ArmContainer,
    berechne_verbindungslinien,
)

# Output handling
from output_handling import (
    save_to_csv,
    plot_ha_with_trend_arms,
    dump_plot_arms_to_txt,
    generate_plot_arms,
    save_ha_kerzen_csv,
)
# --- KONFIGURATION ---
TICKER = "ETH-USD"
INTERVAL = "1m"
HOURS_TO_ANALYZE = 2
# Beispiele:
# TICKER = "EURUSD=X"
# TICKER = "NVDA"
# INTERVAL = "1m"
# HOURS_TO_ANALYZE = 3

# Wohin wir alles schreiben:
BASE_OUTPUT_DIR = os.environ.get("TREND_OUTPUT_DIR", r"D:\TradingBot\output")

def main():
    print("\n" + "=" * 60)
    print("HEIKIN-ASHI TRENDARM-ANALYSE".center(60))
    print(f"Ticker: {TICKER} | Intervall: {INTERVAL}".center(60))
    print("=" * 60 + "\n")

    try:
        # Output-Verzeichnisse sicherstellen
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(r"D:\TradingBot\output", exist_ok=True)

        # 1) Daten herunterladen und vorbereiten
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=HOURS_TO_ANALYZE)

        raw_data = download_data(TICKER, start_date, end_date, INTERVAL)
        if raw_data.empty:
            raise ValueError("Keine Rohdaten von yfinance erhalten. Prüfen Sie Ticker, Internet oder API-Limits.")
        print(f"✅ Rohdaten erfolgreich heruntergeladen. {len(raw_data)} Kerzen.")

        original_data = prepare_data(raw_data)
        if original_data.empty:
            raise ValueError("Rohdaten konnten nach der Vorbereitung nicht verwendet werden.")
        print(f"✅ Rohdaten erfolgreich vorbereitet. {len(original_data)} Kerzen verbleiben.")

        # 2) HA bauen + HA-only kanonisieren
        ha_df = calculate_ha(original_data)
        if {"HA_Open", "HA_High", "HA_Low", "HA_Close"}.issubset(ha_df.columns):
            ha_df = canonicalize_to_ha(ha_df)
        else:
            ha_df.attrs["data_mode"] = "HA"  # Guard zufriedenstellen
        # 3) Cleanup: isolierte & Mikro-Brücken-Kerzen (Marktrauschen) entfernen
        try:
            pre_iso = count_isolated_ha_candles(ha_df)
        except Exception:
            pre_iso = None

        ha_df = remove_isolated_candles(ha_df)
        ha_df = remove_micro_flip_candles(
            ha_df,
            body_ratio=0.20,   # ≤20% des Nachbar-Körperdurchschnitts
            range_ratio=0.50,  # ≤50% der Nachbar-Range
            verbose=True
        )

        try:
            post_iso = count_isolated_ha_candles(ha_df)
        except Exception:
            post_iso = None

        if pre_iso is not None and post_iso is not None:
            print(f"🧹 Isolierte vor Cleanup: {pre_iso}, nach Cleanup: {post_iso}")
        print(f"✅ HA-Daten nach Cleanup: {len(ha_df)} Kerzen.")

        # (optional) Heikin-Ashi-Farbe, falls nicht vorhanden
        if "Farbe" not in ha_df.columns and "Trend" in ha_df.columns:
            ha_df["Farbe"] = bestimme_heikin_ashi_farbe(ha_df)

        # 4) Trendarme erkennen (auf bereinigten HA-Daten!)
        arms = detect_trend_arms(ha_df)
        print(f"✅ {len(arms)} Trendarme erkannt.")

        # 5) Container bauen & validieren
        print("ℹ️ Starte Validierung der Trendarme auf Heikin-Ashi Daten...")
        arm_container = ArmContainer(debug_mode=True)
        for arm in arms:
            arm_container.add_arm(arm)

        arm_container.validate_arms(ha_df)
        validated_arms = [arm for arm in arm_container.arms if getattr(arm, "validated", False)]
        print(f"✅ Validierung abgeschlossen. Validierte Arme: {len(validated_arms)}")

        # 6) Verbindungslinien (C-Serie) berechnen & Plot-Arms erzeugen
        verbindungen_liste = berechne_verbindungslinien(validated_arms, ha_df)
        print("\n[DEBUG] Verbindungslinien (C-Serie):")
        for idx, v in enumerate(verbindungen_liste, start=1):
            print(f"C{idx}: {v}")

        arm_container.plot_arms = generate_plot_arms(verbindungen_liste, ha_df)

        # 7) Plot erstellen (B-/A-Serien auf Wunsch ein-/ausblenden)
        fig = plot_ha_with_trend_arms(
            ha_data=ha_df,
            arm_container=arm_container,
            ticker=TICKER,
            interval=INTERVAL,
            show_plot_a=False,   # A-Serie ausblenden
            show_plot_b=False,   # B-Serie ausblenden
            show_plot_c=True     # C-Serie anzeigen
        )

        # 8) Chart speichern – nur wenn wirklich was drauf ist
        chart_path = os.path.join(
            BASE_OUTPUT_DIR, f"HA_Chart_{TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

        if fig is None:
            fig = plt.gcf()

        if fig is not None and fig.axes:
            ax0 = fig.axes[0]
            drew_anything = not (
                len(ax0.lines) == 0 and len(ax0.patches) == 0 and len(ax0.collections) == 0
            )
            if drew_anything:
                fig.savefig(chart_path, dpi=300, bbox_inches="tight")
                print(f"✅ Chart gespeichert: {chart_path}")
            else:
                print("⚠️ Nichts gezeichnet – PNG nicht gespeichert (siehe [PlotDBG] Ausgabe).")
        else:
            print("⚠️ Kein Figure-Handle verfügbar.")


        # 9) Exporte
        csv_path = save_to_csv(ha_df, arm_container, BASE_OUTPUT_DIR, TICKER)
        csv_path_ha_kerzen = save_ha_kerzen_csv(ha_df, r"D:\TradingBot\output", TICKER)

        print("\n✅ Analyse erfolgreich abgeschlossen")
        print(f"- Heikin-Ashi CSV: {os.path.abspath(csv_path)}")
        print(f"- HA-Kerzen CSV:  {os.path.abspath(csv_path_ha_kerzen)}")
        print(f"- Chart (PNG):    {os.path.abspath(chart_path)}")

        # 10) Dump der Plot-Arms
        dump_plot_arms_to_txt(arm_container.plot_arms)

        # Optional anzeigen (nur interaktiv)
        plt.show()

    except Exception as e:
        print(f"\n❌ FEHLER: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nEmpfohlene Maßnahmen:")
        print("1. Internetverbindung/VPN prüfen.")
        print("2. Ticker auf BTC-USD oder ein anderes gängiges Symbol testen.")
        print("3. Das angefragte Zeitintervall könnte zu groß/klein sein für yfinance.")
        print("4. Stellen Sie sicher, dass Ihr Systemdatum korrekt eingestellt ist.")


if __name__ == "__main__":
    main()
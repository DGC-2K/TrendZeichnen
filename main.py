#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importe der benötigten Module
from data_processing import download_data, prepare_data, bestimme_heikin_ashi_farbe
from trend_calculation import calculate_ha, detect_trend_arms, ArmContainer, berechne_verbindungslinien
from output_handling import save_to_csv, plot_ha_with_trend_arms, BASE_OUTPUT_DIR
from output_handling import dump_plot_arms_to_txt, generate_plot_arms, save_ha_kerzen_csv
from workflow_pipeline import workflow_pipeline

import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- KONFIGURATION ---
TICKER = "EURUSD=X"
INTERVAL = "1m"
HOURS_TO_ANALYZE = 6

#TICKER = "ETH-USD"
#TICKER = "EURUSD=X"
#TICKER = "NVDA"
#INTERVAL = "1m"
#HOURS_TO_ANALYZE = 3

def main():
    print("\n" + "="*60)
    print(f"HEIKIN-ASHI TRENDARM-ANALYSE".center(60))
    print(f"Ticker: {TICKER} | Intervall: {INTERVAL}".center(60))
    print("="*60 + "\n")
    
    try:
        # Output-Verzeichnisse sicherstellen
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(r"D:\TradingBot\output", exist_ok=True)

        # 1. Daten herunterladen und vorbereiten
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=HOURS_TO_ANALYZE)
        
        raw_data = download_data(TICKER, start_date, end_date, INTERVAL)
        
        if raw_data.empty:
            raise ValueError("Keine Rohdaten von yfinance erhalten. Überprüfen Sie Ticker, Internet oder API-Limits.")
        
        print(f"✅ Rohdaten erfolgreich heruntergeladen. {len(raw_data)} Kerzen.")
        
        original_data = prepare_data(raw_data)
        
        if original_data.empty:
            raise ValueError("Rohdaten konnten nach der Vorbereitung nicht verwendet werden.")
            
        print(f"✅ Rohdaten erfolgreich vorbereitet. {len(original_data)} Kerzen verbleiben.")

        # 2. Zentrale Pipeline aufrufen (inklusive Debug-Ausgaben)
        print("⏳ Starte zentrale Workflow-Pipeline (inkl. Kontroll-CSV-Ausgaben)...")
        ha_data, arms = workflow_pipeline(original_data)

        if ha_data.empty:
            raise ValueError("Heikin-Ashi Daten konnten nicht berechnet werden.")

        print(f"✅ Heikin-Ashi Daten erfolgreich berechnet. {len(ha_data)} HA-Kerzen.")
        print(f"✅ {len(arms)} Trendarme erkannt.")

        # Optional: Heikin-Ashi-Farbe berechnen (falls benötigt)
        if 'Farbe' not in ha_data.columns:
            ha_data['Farbe'] = bestimme_heikin_ashi_farbe(ha_data)

        # 3. Trendarme in ArmContainer
        arm_container = ArmContainer(debug_mode=True)
        for arm_obj in arms:
            arm_container.add_arm(arm_obj)
        
        # 4. Validierung der Trendarme
        print(f"ℹ️ Starte Validierung der Trendarme auf Heikin-Ashi Daten...")
        arm_container.validate_arms(ha_data) 
        validated_arms = [arm for arm in arm_container.arms if arm.validated]
        print(f"✅ Validierung abgeschlossen. Ergebnisse in 'output/arm_validation_debug.txt'.")

        # 5. Verbindungslinien berechnen
        verbindungen_liste = berechne_verbindungslinien(validated_arms, ha_data)

        print("\n[DEBUG] Verbindungslinien (C-Serie):")
        for idx, verbindung in enumerate(verbindungen_liste, start=1):
            print(f"C{idx}: {verbindung}")

        # 6. Plot-Arms generieren & zuweisen
        arm_container.plot_arms = generate_plot_arms(verbindungen_liste, ha_data)

        # 7. Chart plotten
        fig = plot_ha_with_trend_arms(
            ha_data=ha_data,
            arm_container=arm_container,
            ticker=TICKER,
            interval=INTERVAL,
            show_plot_a=False,  # Original-Trendarme (A1, A2,...) ausblenden
            show_plot_b=False,  # Validierte Trendarme (B1, B2,...) ausblenden
            show_plot_c=True    # Nur Verbindungslinien (C1, C2,...) anzeigen
        )
        
        # 8. Speichern des Charts
        chart_path = os.path.join(BASE_OUTPUT_DIR, f"HA_Chart_{TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')

        csv_path = save_to_csv(ha_data, arm_container, BASE_OUTPUT_DIR, TICKER)
        csv_path_ha_kerzen = save_ha_kerzen_csv(ha_data, r"D:\TradingBot\output", TICKER)
        print(f"\n✅ Analyse erfolgreich abgeschlossen")
        print(f"- Heikin-Ashi CSV: {os.path.abspath(csv_path)}")
        print(f"- Chart (PNG): {os.path.abspath(chart_path)}")

        # 9. Dump der Plot-Arms
        dump_plot_arms_to_txt(arm_container.plot_arms)
        plt.show()

    except Exception as e:
        print(f"\n❌ FEHLER: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nEmpfohlene Maßnahmen:")
        print("1. Internetverbindung/VPN prüfen.")
        print("2. Ticker auf BTC-USD oder ein anderes gängiges Symbol testen.")
        print("3. Das angefragte Zeitintervall könnte zu groß oder zu klein sein für yfinance.")
        print("4. Stellen Sie sicher, dass Ihr Systemdatum korrekt eingestellt ist.")


if __name__ == "__main__":
    main()
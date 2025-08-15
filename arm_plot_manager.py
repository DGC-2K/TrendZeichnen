# -*- coding: utf-8 -*-

import datetime
from trend_calculation import berechne_verbindungslinien, ArmConnection

class ArmPlotManager:
    def __init__(self, debug_path: str = r"D:\TradingBot\output\C-Serie-Debug-Ausgaben4.txt"):
        self.plot_arms = []
        self.debug_path = debug_path

    def update_plot_arms(self, validated_arms, ha_data):
        # Verbindungslinien berechnen
        verbindungen_liste = berechne_verbindungslinien(validated_arms, ha_data)
        self.plot_arms = []

        # Debug-Eingangszustand
        with open(self.debug_path, "a", encoding="utf-8") as dbgfile:
            dbgfile.write(f"\n[INPUT] Validated Arms fÃ¼r update_plot_arms ({datetime.datetime.now().isoformat()}):\n")
            for i, v in enumerate(validated_arms):
                dbgfile.write(f"  B{i+1}: {v.start_idx}-{v.end_idx}, Richtung: {v.direction}, validated: {v.validated}\n")
            dbgfile.write("-" * 50 + "\n")

        # Verbindungslinien in Plot-Arms umwandeln
        for verbindung in verbindungen_liste:
            start_idx, start_price = verbindung['start']
            end_idx, end_price = verbindung['ende']
            direction = 'UP' if end_price > start_price else 'DOWN'

            arm = ArmConnection(
                arm_num=len(self.plot_arms) + 1,
                direction=direction,
                start_idx=start_idx,
                end_idx=end_idx,
                start_price=start_price,
                end_price=end_price,
                validated=True
            )

            # Optional: Mitte fÃ¼r spezielle Verbindungstypen
            if 'mitte' in verbindung:
                mid_idx, mid_price = verbindung['mitte']
                if direction == 'UP':
                    arm.set_pt_data(mid_price, mid_idx)
                else:
                    arm.set_ph_data(mid_price, mid_idx)

            self.plot_arms.append(arm)

        # Debug-Ausgabe
        with open(self.debug_path, "a", encoding="utf-8") as dbgfile:
            dbgfile.write(f"\n[OUTPUT] Plot Arms nach update_plot_arms ({datetime.datetime.now().isoformat()}):\n")
            for i, arm in enumerate(self.plot_arms):
                dbgfile.write(
                    f"  C{i+1}: Kerzen {arm.start_idx}-{arm.end_idx}, Richtung: {arm.direction}, "
                    f"StartPreis: {arm.start_price}, EndPreis: {arm.end_price}, validated: {arm.validated}"
                )
                if hasattr(arm, "pt_candle_idx") and arm.pt_candle_idx is not None:
                    dbgfile.write(f", PT: {arm.pt_candle_idx} ({arm.pt_price if hasattr(arm, 'pt_price') else ''})")
                if hasattr(arm, "ph_candle_idx") and arm.ph_candle_idx is not None:
                    dbgfile.write(f", PH: {arm.ph_candle_idx} ({arm.ph_price if hasattr(arm, 'ph_price') else ''})")
                dbgfile.write("\n")
            dbgfile.write("-" * 50 + "\n")

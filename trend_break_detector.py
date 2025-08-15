# -*- coding: utf-8 -*-

import pandas as pd

class TrendBreakDetector:
    @staticmethod
    def detect_trend_break_and_restart(container, ha_data: pd.DataFrame) -> None:
        """
        Untersucht TrendbrÃ¼che und startet neue Trendarme
        :param container: ArmContainer-Instanz
        :param ha_data: Heikin-Ashi Daten
        """
        debug_path = r"D:\TradingBot\output\C-Serie-Debug-Ausgaben200.txt"
        
        # === PRÃœFCODE BEGINN ===
        with open(debug_path, "a", encoding="utf-8") as dbgfile:
            dbgfile.write("\n--- PRÃœFCODE: Plot-Arms (C-Serie) ---\n")
            for idx, c_arm in enumerate(container.plot_arms, start=1):
                dbgfile.write(
                    f"  C{idx}: start_idx={c_arm.start_idx}, end_idx={c_arm.end_idx}, Richtung: {c_arm.direction}, broken: {c_arm.broken}, validated: {c_arm.validated}\n"
                )
            dbgfile.write(f"Anzahl Plot-Arms: {len(container.plot_arms)}\n")
            dbgfile.write(f"DataFrame-LÃ¤nge: {len(ha_data)}; Indizes: {ha_data.index[0]} bis {ha_data.index[-1]}\n")
            dbgfile.write("--- PRÃœFCODE ENDE ---\n")
        # === PRÃœFCODE ENDE ===

        # Debug-Ausgabe fÃ¼r den aktuellen Stand der C-Serie
        with open(debug_path, "a", encoding="utf-8") as dbgfile:
            import datetime
            dt = datetime.datetime.now()
            dbgfile.write("\n### NEUER DEBUG BLOCK AKTIV ###\n")
            dbgfile.write(f"[C] Stand der berechneten C-Serie ({dt.strftime('%Y-%m-%dT%H:%M:%S.%f')}):\n")
            if not container.plot_arms:
                dbgfile.write("  [Keine C-Serie vorhanden]\n")
            else:
                for idx, c_arm in enumerate(container.plot_arms, start=1):
                    dbgfile.write(
                        f"  C{idx}: Kerzen {c_arm.start_idx}-{c_arm.end_idx}, "
                        f"Richtung: {c_arm.direction}, "
                        f"Status: {'broken' if c_arm.broken else 'ok'}, "
                        f"validated: {c_arm.validated}\n"
                    )
            dbgfile.write("-" * 50 + "\n")

        # Arbeite nur auf der aktuellen C-Serie (Plot-Arms)
        c_arms = [arm for arm in container.plot_arms if not arm.broken]
        if not c_arms or ha_data.empty:
            return

        # Finde das letzte validierte C-Arm
        last_validated_idx = None
        for i in reversed(range(len(c_arms))):
            if c_arms[i].validated:
                last_validated_idx = i
                break

        if last_validated_idx is None:
            return

        last_validated_arm = c_arms[last_validated_idx]

        # Bounds bestimmen aus den Kerzen des letzten validierten C-Armes
        if last_validated_arm.end_idx >= len(ha_data):
            return

        arm_data = ha_data.iloc[last_validated_arm.start_idx:last_validated_arm.end_idx + 1]
        upper_bound = float(arm_data['High'].max())
        lower_bound = float(arm_data['Low'].min())

        if last_validated_arm.direction == "DOWN":
            bound_value = upper_bound
            cmp_op = lambda close: close > bound_value
            new_trend_direction = "UP"
        elif last_validated_arm.direction == "UP":
            bound_value = lower_bound
            cmp_op = lambda close: close < bound_value
            new_trend_direction = "DOWN"
        else:
            return

        # PrÃ¼fe ALLE Close-Werte nach dem letzten validierten C-Arm
        candle_start = last_validated_arm.end_idx + 1
        closes = [float(ha_data.iloc[i]['Close']) for i in range(candle_start, len(ha_data))]
        debug_closes = closes[:20]

        with open(debug_path, "a", encoding="utf-8") as dbgfile:
            dbgfile.write(f"Letzter validierter C-Arm endet bei Index {last_validated_arm.end_idx}\n")
            dbgfile.write(f"Bounds: upper={upper_bound}, lower={lower_bound}\n")
            dbgfile.write(f"Richtung: {last_validated_arm.direction}\n")
            dbgfile.write(f"Untersuchte Close-Werte (erste 20): {debug_closes}\n")

        found_break = False
        break_index = None
        for i in range(0, len(closes) - 2):
            trio = closes[i:i + 3]
            if all(cmp_op(c) for c in trio):
                found_break = True
                break_index = i
                break

        with open(debug_path, "a", encoding="utf-8") as dbgfile:
            dbgfile.write(f"Trendbruch gefunden: {found_break} an Trio-Index {break_index}\n")

        if found_break:
            # Markiere alle aktuellen Plot-Arms als "broken"
            for arm in c_arms:
                arm.broken = True
            container.broken_arms.extend(c_arms)
            container.arms = []
            container.plot_arms = []

            # Neuer Arm startet am Endpunkt des letzten validierten Plot-Arms
            new_start_idx = last_validated_arm.end_idx
            new_start_price = last_validated_arm.end_price

            new_arm = ArmConnection(
                arm_num=1,
                direction=new_trend_direction,
                start_idx=new_start_idx,
                end_idx=new_start_idx,
                start_price=new_start_price,
                end_price=new_start_price,
                validated=False
            )
            new_arm.broken = False
            container.add_arm(new_arm)

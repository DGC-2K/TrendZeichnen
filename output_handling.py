import os
from datetime import datetime, timedelta
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MinuteLocator

from trend_calculation import (
    ArmConnection,
    ArmContainer,
    berechne_verbindungslinien,
    find_max_high_in_range,
    find_min_low_in_range,
)

BASE_OUTPUT_DIR = "D:\\TradingBot\\output"

def debug_verbindungen_liste(verbindungen_liste, serie_typ, file_path, arm_container=None):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== Debug-Ausgabe für {serie_typ} =====\n")
        for idx, v in enumerate(verbindungen_liste):
            f.write(f"Verbindung {idx}: Typ={v.get('typ')}, ")
            f.write(f"Start={v.get('start')}, ")
            if "mitte" in v:
                f.write(f"Mitte={v.get('mitte')}, ")
            f.write(f"Ende={v.get('ende')}\n")
        f.write("===== Ende Debug-Ausgabe =====\n")

        # --- NEU: Bounds anhängen (falls vorhanden) ---
        if arm_container is not None and getattr(arm_container, "debug_mode", True):
            b = arm_container.bounds
            src = getattr(arm_container, "bounds_source_arm_num", 0)
            # Versuche, die letzte validierte Plot-Arm-Nummer zu ermitteln:
            try:
                last_valid = None
                for i, a in enumerate(getattr(arm_container, "plot_arms", []), start=1):
                    if getattr(a, "validated", False):
                        last_valid = i
                if last_valid is not None:
                    f.write("\n[OUTPUT] Plot-Arm-Bounds:\n")
                    f.write(
                        f"C{last_valid}_BOUNDS: upper={float(b.upper):.5f}, "
                        f"lower={float(b.lower):.5f}, source_arm={int(src)}, validated=True\n"
                    )
            except Exception:
                pass


def generate_plot_arms(verbindungen_liste, ha_data, debug_file=None, serie_typ="C-Serie", arm_container=None) -> List[ArmConnection]:
    debug_file = os.path.join(BASE_OUTPUT_DIR, "C-Serie-Debug-Ausgaben5.txt")
    debug_verbindungen_liste(verbindungen_liste, serie_typ + " (PlotArms Eingang)", debug_file, arm_container=arm_container)

    plot_arms = []
    for v in verbindungen_liste:
        if v['typ'] == 'B1':
            start_idx, start_price = v['start']
            end_idx, end_price = v['ende']
            direction = 'UP' if end_price > start_price else 'DOWN'
            arm = ArmConnection(
                arm_num=len(plot_arms) + 1,
                direction=direction,
                start_idx=start_idx,
                end_idx=end_idx,
                start_price=start_price,
                end_price=end_price,
                validated=True
            )
            plot_arms.append(arm)
        elif v['typ'] == 'B-C':
            start_idx, start_price = v['start']
            end_idx, end_price = v['ende']
            direction = 'UP' if end_price > start_price else 'DOWN'
            arm = ArmConnection(
                arm_num=len(plot_arms) + 1,
                direction=direction,
                start_idx=start_idx,
                end_idx=end_idx,
                start_price=start_price,
                end_price=end_price,
                validated=True
            )
            plot_arms.append(arm)
        elif v['typ'] == 'B-D-C':
            start_idx, start_price = v['start']
            mitte_idx, mitte_price = v['mitte']
            end_idx, end_price = v['ende']
            direction1 = 'UP' if mitte_price > start_price else 'DOWN'
            direction2 = 'UP' if end_price > mitte_price else 'DOWN'
            arm1 = ArmConnection(
                arm_num=len(plot_arms) + 1,
                direction=direction1,
                start_idx=start_idx,
                end_idx=mitte_idx,
                start_price=start_price,
                end_price=mitte_price,
                validated=True
            )
            arm2 = ArmConnection(
                arm_num=len(plot_arms) + 1,
                direction=direction2,
                start_idx=mitte_idx,
                end_idx=end_idx,
                start_price=mitte_price,
                end_price=end_price,
                validated=True
            )
            plot_arms.append(arm1)
            plot_arms.append(arm2)
    # Debug-Ausgabe in Datei schreiben (PlotArms)
    with open(debug_file, "a", encoding="utf-8") as f:
        f.write("\n[OUTPUT] Plot Arms nach generate_plot_arms:\n")
        for i, arm in enumerate(plot_arms):
            f.write(
                f"  C{i+2}: Kerzen {arm.start_idx}-{arm.end_idx}, Richtung: {arm.direction}, "
                f"StartPreis: {arm.start_price:.5f}, EndPreis: {arm.end_price:.5f}, "
                f"validated: {arm.validated}"
            )
            f.write("\n")
        f.write("-" * 50 + "\n")

    return plot_arms


def plot_ha_with_trend_arms(
    ha_data, arm_container, ticker, interval,
    show_plot_a=True, show_plot_b=True, show_plot_c=True
):
    if 'Zeit' not in ha_data.columns:
        raise ValueError("Die Spalte 'Zeit' muss im ha_data DataFrame vorhanden sein.")
    ha_data['Zeit'] = pd.to_datetime(ha_data['Zeit'])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_title(f"Heikin Ashi Chart - {ticker} - Interval: {interval}", 
                fontsize=16, fontweight='bold', pad=10)
    
    dates = mdates.date2num(np.array(ha_data['Zeit'].dt.to_pydatetime()))
    interval_to_minutes = {'1m': 1, '2m': 2, '5m': 5, '15m': 15, 
                          '30m': 30, '60m': 60, '1h': 60, '1d': 1440}
    minutes_per_interval = interval_to_minutes.get(interval, 2)
    width = (minutes_per_interval / (24 * 60)) * 0.70

    # Heikin Ashi Kerzen plotten
    for i in range(len(ha_data)):
        row = ha_data.iloc[i]
        open_val, close_val = float(row['Open']), float(row['Close'])
        candle_color = 'limegreen' if row['Trend'] == 'UP' else 'red'
        ax.bar(dates[i], abs(close_val - open_val), bottom=min(open_val, close_val),
               width=width, color=candle_color, edgecolor='black', linewidth=0.5, zorder=2)
        ax.vlines(dates[i], float(row['Low']), float(row['High']), 
                 color='black', linewidth=0.8, zorder=1)
        # Offset relativ zur Preisspanne statt Multiplikation, und innerhalb clippen
        price_range = float(ha_data['High'].max() - ha_data['Low'].min()) or 1.0
        ax.text(dates[i], float(row['High']) + 0.01 * price_range, str(ha_data.index[i]),
                ha='center', va='bottom', fontsize=8, color='black', zorder=3, clip_on=True)

    # --- Originelle Trendarme: A1, A2, ... ---
    if show_plot_a and hasattr(arm_container, 'arms'):
        for arm in arm_container.arms:
            if 0 <= arm.start_idx < len(ha_data) and 0 <= arm.end_idx < len(ha_data):
                x_coords = [dates[arm.start_idx], dates[arm.end_idx]]
                y_coords = [arm.start_price, arm.end_price]
                ax.plot(x_coords, y_coords, color='magenta', linewidth=2.5, zorder=10)
                mid_x = (x_coords[0] + x_coords[1]) / 2
                mid_y = (y_coords[0] + y_coords[1]) / 2
                label = f"A{arm.arm_num}"
                ax.text(mid_x, mid_y, label, fontsize=13, color='magenta', 
                        fontweight='bold', zorder=11)

    # --- Validierte Trendarme: B1, B2, ... ---
    if show_plot_b:
        validated_arms = [arm for arm in getattr(arm_container, "arms", []) 
                         if getattr(arm, "validated", False)]
        for v_idx, arm in enumerate(validated_arms, start=1):
            if 0 <= arm.start_idx < len(ha_data) and 0 <= arm.end_idx < len(ha_data):
                x_coords = [dates[arm.start_idx], dates[arm.end_idx]]
                y_coords = [arm.start_price, arm.end_price]
                ax.plot(x_coords, y_coords, color='black', linewidth=2, 
                       linestyle='--', zorder=10)
                mid_x = (x_coords[0] + x_coords[1]) / 2
                mid_y = (y_coords[0] + y_coords[1]) / 2
                label = f"B{v_idx}"
                ax.text(mid_x, mid_y, label, fontsize=13, color='black', 
                        fontweight='bold', zorder=11)

    # --- Verbindungslinien aus validierten Armen: C1, C2, ... ---
    if show_plot_c:
        validated_arms = [arm for arm in getattr(arm_container, "arms", []) 
                          if getattr(arm, "validated", False)]
        if validated_arms:
            verbindungen_liste = berechne_verbindungslinien(validated_arms, ha_data)
            plot_arms = generate_plot_arms(verbindungen_liste, ha_data)
            verbindung_count = 1
            for arm in plot_arms:
                if 0 <= arm.start_idx < len(ha_data) and 0 <= arm.end_idx < len(ha_data):
                    x_coords = [dates[arm.start_idx], dates[arm.end_idx]]
                    y_coords = [arm.start_price, arm.end_price]
                    ax.plot(x_coords, y_coords, color='blue', linewidth=1.2, linestyle=':', zorder=9)
                    mid_x = (x_coords[0] + x_coords[1]) / 2
                    mid_y = (y_coords[0] + y_coords[1]) / 2
                    label = f"C{verbindung_count}"
                    ax.text(mid_x, mid_y, label, fontsize=11, color='blue', 
                            fontweight='bold', zorder=11)
                    verbindung_count += 1

    # --- Fibonacci 38,2% Linien je Arm ---
    try:
        arms_for_fib = [arm for arm in getattr(arm_container, "arms", []) if getattr(arm, "validated", False)]
        if not arms_for_fib:
            arms_for_fib = list(getattr(arm_container, "arms", []))

        for arm in arms_for_fib:
            if not (0 <= arm.start_idx < len(ha_data) and 0 <= arm.end_idx < len(ha_data)):
                continue
            arm_slice = ha_data.iloc[arm.start_idx:arm.end_idx + 1]
            if arm_slice.empty or not {'High','Low'}.issubset(arm_slice.columns):
                continue
            arm_high = float(arm_slice['High'].max())
            arm_low  = float(arm_slice['Low'].min())
            span = arm_high - arm_low
            if span <= 0:
                continue
            if arm.direction == 'UP':
                fib382 = arm_high - 0.382 * span
            elif arm.direction == 'DOWN':
                fib382 = arm_low + 0.382 * span
            else:
                continue
            x_start = dates[arm.start_idx]
            x_end   = dates[arm.end_idx]
            ax.hlines(fib382, x_start, x_end, linestyles='dashdot', linewidth=1.6, color='orange', zorder=8)
            ax.text(x_end, fib382, "38,2%", va='bottom', ha='right', fontsize=9, color='orange', zorder=9)
    except Exception:
        pass

    # Restliche Plot-Einstellungen
    ax.xaxis.set_major_locator(MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel("Zeit", fontsize=12, labelpad=10)
    ax.set_ylabel("Preis", fontsize=12, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Achsenskalierung wie gehabt
    all_prices = []
    if not ha_data.empty:
        all_prices.extend(ha_data['High'].tolist())
        all_prices.extend(ha_data['Low'].tolist())
    if show_plot_a and hasattr(arm_container, 'arms'):
        for arm in arm_container.arms:
            if arm.start_idx < len(ha_data) and arm.end_idx < len(ha_data):
                all_prices.append(arm.start_price)
                all_prices.append(arm.end_price)
    if show_plot_b:
        for arm in validated_arms:
            if arm.start_idx < len(ha_data) and arm.end_idx < len(ha_data):
                all_prices.append(arm.start_price)
                all_prices.append(arm.end_price)
    all_prices = [p for p in all_prices if not np.isnan(p)]
    if all_prices:
        min_price = min(all_prices)
        max_price = max(all_prices)
        price_range = max_price - min_price
        padding = price_range * 0.1 if price_range != 0 else max(min_price * 0.005, 0.5)
        ax.set_ylim(min_price - padding, max_price + padding)
    else:
        ax.set_ylim(0, 100)

    # X-Achsen-Skalierung
    if len(dates) > 1:
        x_min_val = dates.min()
        x_max_val = dates.max()
        x_range = x_max_val - x_min_val
        ax.set_xlim(x_min_val - x_range * 0.05, x_max_val + x_range * 0.05)
    elif len(dates) == 1:
        padding = (minutes_per_interval / (60 * 24)) * 5
        ax.set_xlim(dates[0] - padding, dates[0] + padding)
    else:
        ax.set_xlim(mdates.date2num(datetime.now() - timedelta(hours=1)), 
                   mdates.date2num(datetime.now()))

    # Plot speichern
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    plot_filename = os.path.join(BASE_OUTPUT_DIR, 
                                f"heikin_ashi_trend_arms_{ticker}_{interval}.png")
    try:
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
    except Exception as e:
        print(f"Fehler beim Speichern des Plots: {e}")

    return fig

def save_to_csv(ha_data: pd.DataFrame, arm_container: ArmContainer, output_dir_param: str, ticker: str) -> str:
    os.makedirs(output_dir_param, exist_ok=True)
    
    def german_format(x):
        try:
            return f"{float(x):,.5f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (ValueError, TypeError):
            if x is not None and str(x).strip() != '':
                return "0,000"
            return ''
            
    if 'Zeit' not in ha_data.columns:
        raise ValueError("Die Spalte 'Zeit' muss im ha_data DataFrame vorhanden sein, um die CSV korrekt zu formatieren.")
    
    ha_data['Zeit'] = pd.to_datetime(ha_data['Zeit'])

    csv_data = pd.DataFrame({
        'Kerze_Nr': ha_data.index.values,
        'Zeit': ha_data['Zeit'].dt.strftime('%d.%m.%Y %H:%M'),
        'Open': ha_data['Open'].apply(german_format),
        'High': ha_data['High'].apply(german_format),
        'Low': ha_data['Low'].apply(german_format),
        'Close': ha_data['Close'].apply(german_format),
        'Trend': ha_data['Trend'],
        'Arm_Nr': '',
        'Arm_Richtung': '',
        'Arm_Startpreis': '',
        'Arm_Endpreis': '',
        'Validiert': ''
    })

    for arm in arm_container.arms:
        start_idx_in_csv = arm.start_idx
        end_idx_in_csv = arm.end_idx

        if start_idx_in_csv >= 0 and end_idx_in_csv < len(csv_data) and start_idx_in_csv <= end_idx_in_csv:
            idx_slice = slice(start_idx_in_csv, end_idx_in_csv + 1)
            
            csv_data.iloc[idx_slice, csv_data.columns.get_loc('Arm_Nr')] = arm.arm_num
            csv_data.iloc[idx_slice, csv_data.columns.get_loc('Arm_Richtung')] = arm.direction
            csv_data.iloc[idx_slice, csv_data.columns.get_loc('Arm_Startpreis')] = german_format(arm.start_price)
            csv_data.iloc[idx_slice, csv_data.columns.get_loc('Arm_Endpreis')] = german_format(arm.end_price)
            csv_data.iloc[idx_slice, csv_data.columns.get_loc('Validiert')] = 'Ja' if arm.validated else 'Nein'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir_param, f"HA_Trendarme_{ticker}_{timestamp}.csv")
    
    csv_data.to_csv(csv_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"Daten und Trendarme in {csv_path} gespeichert.")
    return csv_path

def save_ha_kerzen_csv(ha_data: pd.DataFrame, output_dir_param: str, ticker: str) -> str:
    os.makedirs(output_dir_param, exist_ok=True)

    if 'Zeit' not in ha_data.columns:
        raise ValueError("Die Spalte 'Zeit' muss im ha_data DataFrame vorhanden sein, um die CSV korrekt zu formatieren.")

    ha_data = ha_data.copy()
    ha_data['Zeit'] = pd.to_datetime(ha_data['Zeit'])

    csv_data = pd.DataFrame({
        'Kerze_Nr': ha_data.index.values,
        'Zeit': ha_data['Zeit'].dt.strftime('%d.%m.%Y %H:%M'),
        # WICHTIG: numerisch lassen, NICHT über german_format in Strings umwandeln
        'Open':  pd.to_numeric(ha_data['Open'],  errors='coerce'),
        'High':  pd.to_numeric(ha_data['High'],  errors='coerce'),
        'Low':   pd.to_numeric(ha_data['Low'],   errors='coerce'),
        'Close': pd.to_numeric(ha_data['Close'], errors='coerce'),
    })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir_param, f"HA_Kerzen_{ticker}_{timestamp}.csv")

    # Feste Präzision für alle Floats; Dezimalkomma für DE
    csv_data.to_csv(
        csv_path,
        index=False,
        sep=';',
        decimal=',',
        encoding='utf-8-sig',
        float_format='%.5f'
    )
    print(f"HA-Kerzen wurden in {csv_path} gespeichert.")
    return csv_path


def dump_plot_arms_to_txt(plot_arms: List[ArmConnection], file_path: str = "output/plot_arms_debug.txt"):
    import datetime
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"\n[INPUT] Validated Arms für update_plot_arms ({datetime.datetime.now().isoformat()}):\n")
        for i, arm in enumerate(plot_arms):
            f.write(f"  C{i+1}: {arm.start_idx}-{arm.end_idx}, Richtung: {arm.direction}, validated: {arm.validated}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Anzahl Plot-Arms: {len(plot_arms)}\n")

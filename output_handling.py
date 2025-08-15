# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MinuteLocator

from trend_calculation import (
    ArmConnection,
    ArmContainer,
    berechne_verbindungslinien,
    find_max_high_in_range,
    find_min_low_in_range,
    build_color_runs,
    annotate_dow_per_run,
    annotate_arms_with_runs,
)

# --------------------------------------------------------------------
# Hilfsfunktionen für Verbindungs-Debug & Plot-Arm-Erzeugung
# --------------------------------------------------------------------

def debug_verbindungen_liste(verbindungen_liste, serie_typ, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== Debug-Ausgabe für {serie_typ} =====\n")
        for idx, v in enumerate(verbindungen_liste):
            f.write(f"Verbindung {idx}: Typ={v.get('typ')}, ")
            f.write(f"Start={v.get('start')}, ")
            if "mitte" in v:
                f.write(f"Mitte={v.get('mitte')}, ")
            f.write(f"Ende={v.get('ende')}\n")
        f.write("===== Ende Debug-Ausgabe =====\n")


def generate_plot_arms(verbindungen_liste, ha_data, serie_typ: str = "unbekannt") -> List[ArmConnection]:
    debug_file = os.path.join("D:\\TradingBot\\output", "C-Serie-Debug-Ausgaben5.txt")
    os.makedirs(os.path.dirname(debug_file), exist_ok=True)
    debug_verbindungen_liste(verbindungen_liste, serie_typ + " (PlotArms Eingang)", debug_file)

    def _set_fib382(arm: ArmConnection):
        span_low  = min(arm.start_price, arm.end_price)
        span_high = max(arm.start_price, arm.end_price)
        rng = span_high - span_low
        if rng > 0:
            arm.fib382 = (span_high - 0.382 * rng) if arm.direction == 'UP' else (span_low + 0.382 * rng)
        else:
            arm.fib382 = None

    plot_arms: List[ArmConnection] = []
    for v in verbindungen_liste:
        typ = v.get("typ")
        if typ in ("B1", "B-C"):
            start_idx, start_price = v["start"]
            end_idx, end_price = v["ende"]
            direction = "UP" if end_price > start_price else "DOWN"
            arm = ArmConnection(
                arm_num=len(plot_arms) + 1,
                direction=direction,
                start_idx=start_idx,
                end_idx=end_idx,
                start_price=start_price,
                end_price=end_price,
                validated=True,
            )
            _set_fib382(arm)
            plot_arms.append(arm)

        elif typ == "B-D-C":
            start_idx, start_price = v["start"]
            mitte_idx, mitte_price = v["mitte"]
            end_idx, end_price = v["ende"]

            direction1 = "UP" if mitte_price > start_price else "DOWN"
            direction2 = "UP" if end_price > mitte_price else "DOWN"

            arm1 = ArmConnection(
                arm_num=len(plot_arms) + 1,
                direction=direction1,
                start_idx=start_idx,
                end_idx=mitte_idx,
                start_price=start_price,
                end_price=mitte_price,
                validated=True,
            )
            _set_fib382(arm1)

            arm2 = ArmConnection(
                arm_num=len(plot_arms) + 1,
                direction=direction2,
                start_idx=mitte_idx,
                end_idx=end_idx,
                start_price=mitte_price,
                end_price=end_price,
                validated=True,
            )
            _set_fib382(arm2)

            plot_arms.extend([arm1, arm2])

    # Debug-Ausgabe in Datei (PlotArms)
    with open(debug_file, "a", encoding="utf-8") as f:
        f.write("\n[OUTPUT] Plot Arms nach generate_plot_arms:\n")
        for i, arm in enumerate(plot_arms):
            line = (
                f"  C{i+1}: Kerzen {arm.start_idx}-{arm.end_idx}, Richtung: {arm.direction}, "
                f"StartPreis: {arm.start_price:.6f}, EndPreis: {arm.end_price:.6f}, "
                f"validated: {arm.validated}"
            )
            if getattr(arm, "fib382", None) is not None:
                line += f", FIB38: {arm.fib382:.6f}"
            f.write(line + "\n")
        f.write("-" * 50 + "\n")

    return plot_arms

# --------------------------------------------------------------------
# Haupt-Plotfunktion (stabil & ohne Modulebenen-Seiteneffekte)
# --------------------------------------------------------------------

def plot_ha_with_trend_arms(
    ha_data: pd.DataFrame, arm_container: ArmContainer, ticker: str, interval: str,
    show_plot_a: bool = True, show_plot_b: bool = True, show_plot_c: bool = True
):
    # Guards & Aufbereitung
    if 'Zeit' not in ha_data.columns:
        raise ValueError("Die Spalte 'Zeit' muss im ha_data DataFrame vorhanden sein.")
    ha_data = ha_data.copy()
    ha_data['Zeit'] = pd.to_datetime(ha_data['Zeit'])

    for _c in ("Open", "High", "Low", "Close"):
        if _c in ha_data.columns and not pd.api.types.is_float_dtype(ha_data[_c]):
            ha_data[_c] = pd.to_numeric(ha_data[_c], errors="coerce")
    ha_data = ha_data.dropna(subset=["Zeit", "Open", "High", "Low", "Close"]).reset_index(drop=True)

    n = len(ha_data)
    if n == 0:
        raise ValueError("ha_data ist nach Cleanup leer – nichts zu plotten.")

    def _regime_color(regime: str) -> str:
        if regime == "UP":   return "limegreen"
        if regime == "DOWN": return "red"
        return "gray"

    # Figure/Axes
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_title(f"Heikin Ashi Chart - {ticker} - Interval: {interval}",
                 fontsize=16, fontweight='bold', pad=10)

    dates = mdates.date2num(np.array(ha_data['Zeit'].dt.to_pydatetime()))
    interval_to_minutes = {'1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30, '60m': 60, '1h': 60, '1d': 1440}
    minutes_per_interval = interval_to_minutes.get(interval, 2)
    width = (minutes_per_interval / (24 * 60)) * 0.70

    # Runs (Dow-Regel)
    try:
        runs = build_color_runs(ha_data)
        runs = annotate_dow_per_run(runs)
    except Exception as e:
        print(f"Dow-Runs konnten nicht erstellt werden: {e}")
        runs = []

    # Kerzen
    for i in range(n):
        row = ha_data.iloc[i]
        o, c = float(row['Open']), float(row['Close'])
        color = 'limegreen' if row.get('Trend', 'UP') == 'UP' else 'red'
        ax.bar(dates[i], abs(c - o), bottom=min(o, c),
               width=width, color=color, edgecolor='black', linewidth=0.5, zorder=2)
        ax.vlines(dates[i], float(row['Low']), float(row['High']),
                  color='black', linewidth=0.8, zorder=1)
        ax.text(dates[i], float(row['High']) * 1.0005, str(ha_data.index[i]),
                ha='center', va='bottom', fontsize=8, color='black', zorder=3)

    # A-Serie
    if show_plot_a and hasattr(arm_container, 'arms'):
        for arm in arm_container.arms:
            if 0 <= arm.start_idx < n and 0 <= arm.end_idx < n:
                x = [dates[arm.start_idx], dates[arm.end_idx]]
                y = [arm.start_price, arm.end_price]
                ax.plot(x, y, color='magenta', linewidth=2.5, zorder=10)
                ax.text((x[0]+x[1])/2, (y[0]+y[1])/2, f"A{arm.arm_num}",
                        fontsize=13, color='magenta', fontweight='bold', zorder=11)

    # B-Serie
    validated_arms: List[ArmConnection] = []
    if show_plot_b:
        validated_arms = [a for a in getattr(arm_container, "arms", []) if getattr(a, "validated", False)]
        if runs:
            annotate_arms_with_runs(validated_arms, runs)
        for i, arm in enumerate(validated_arms, start=1):
            if 0 <= arm.start_idx < n and 0 <= arm.end_idx < n:
                x = [dates[arm.start_idx], dates[arm.end_idx]]
                y = [arm.start_price, arm.end_price]
                col = _regime_color(getattr(arm, "regime", "RANGE"))
                ax.plot(x, y, color=col, linewidth=2, linestyle='--', zorder=10)
                ax.text((x[0]+x[1])/2, (y[0]+y[1])/2, f"B{i}",
                        fontsize=13, color=col, fontweight='bold', zorder=11)

    # C-Serie
    if show_plot_c:
        if not validated_arms:
            validated_arms = [a for a in getattr(arm_container, "arms", []) if getattr(a, "validated", False)]
        if validated_arms:
            verbindungen = berechne_verbindungslinien(validated_arms, ha_data)
            plot_arms = generate_plot_arms(verbindungen, ha_data)
            if runs:
                annotate_arms_with_runs(plot_arms, runs)

            c_idx = 1
            for arm in plot_arms:
                if 0 <= arm.start_idx < n and 0 <= arm.end_idx < n:
                    x = [dates[arm.start_idx], dates[arm.end_idx]]
                    y = [arm.start_price, arm.end_price]
                    col = _regime_color(getattr(arm, "regime", "RANGE"))
                    ax.plot(x, y, color=col, linewidth=1.6, linestyle=':', zorder=9)

                    level = getattr(arm, "fib382", None)
                    if level is not None:
                        ax.hlines(level, x[0], x[1], linestyles="dashed", linewidth=1,
                                  zorder=12, color='purple')
                        ax.text(x[1], level, "38.2%", va="bottom", ha="left",
                                fontsize=8, zorder=13)

                    ax.text((x[0]+x[1])/2, (y[0]+y[1])/2, f"C{c_idx}",
                            fontsize=11, color=col, fontweight='bold', zorder=11)
                    c_idx += 1

    # Achsen & Layout
    ax.xaxis.set_major_locator(MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel("Zeit", fontsize=12, labelpad=10)
    ax.set_ylabel("Preis", fontsize=12, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    all_prices = []
    if not ha_data.empty:
        all_prices += ha_data['High'].tolist() + ha_data['Low'].tolist()
    if show_plot_a and hasattr(arm_container, 'arms'):
        for a in arm_container.arms:
            if a.start_idx < n and a.end_idx < n:
                all_prices += [a.start_price, a.end_price]
    if show_plot_b:
        for a in validated_arms:
            if a.start_idx < n and a.end_idx < n:
                all_prices += [a.start_price, a.end_price]
    all_prices = [p for p in all_prices if not np.isnan(p)]
    if all_prices:
        mn, mx = min(all_prices), max(all_prices)
        pr = mx - mn
        pad = pr * 0.10 if pr != 0 else max(mn * 0.005, 0.5)
        ax.set_ylim(mn - pad, mx + pad)

    if len(dates) > 1:
        x_min, x_max = dates.min(), dates.max()
        xr = x_max - x_min
        ax.set_xlim(x_min - xr * 0.05, x_max + xr * 0.05)
    elif len(dates) == 1:
        pad = (minutes_per_interval / (60 * 24)) * 5
        ax.set_xlim(dates[0] - pad, dates[0] + pad)

    fig.tight_layout()
    fig.canvas.draw_idle()

    # Debug
    candles = sum(isinstance(p, plt.Rectangle) for p in ax.patches)
    lines = len(ax.lines)
    collections = len(ax.collections)
    print(f"[PlotDBG] candles={candles}, lines={lines}, collections={collections}")

    return fig

# --------------------------------------------------------------------
# CSV/Debug Exporte (sauber auf Funktionen beschränkt)
# --------------------------------------------------------------------

def save_to_csv(ha_data: pd.DataFrame, arm_container: ArmContainer, output_dir_param: str, ticker: str) -> str:
    os.makedirs(output_dir_param, exist_ok=True)

    def german_format(x):
        try:
            return f"{float(x):,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (ValueError, TypeError):
            return "" if (x is None or str(x).strip() == "") else "0,000"

    if 'Zeit' not in ha_data.columns:
        raise ValueError("Die Spalte 'Zeit' muss im ha_data DataFrame vorhanden sein, um die CSV korrekt zu formatieren.")
    ha_data = ha_data.copy()
    ha_data['Zeit'] = pd.to_datetime(ha_data['Zeit'])

    csv_data = pd.DataFrame({
        'Kerze_Nr': ha_data.index.values,
        'Zeit': ha_data['Zeit'].dt.strftime('%d.%m.%Y %H:%M'),
        'Open': ha_data['Open'].apply(german_format),
        'High': ha_data['High'].apply(german_format),
        'Low': ha_data['Low'].apply(german_format),
        'Close': ha_data['Close'].apply(german_format),
        'Trend': ha_data.get('Trend', pd.Series([""] * len(ha_data))),
        'Arm_Nr': '',
        'Arm_Richtung': '',
        'Arm_Startpreis': '',
        'Arm_Endpreis': '',
        'Validiert': ''
    })

    for arm in getattr(arm_container, "arms", []):
        si, ei = arm.start_idx, arm.end_idx
        if 0 <= si <= ei < len(csv_data):
            sl = slice(si, ei + 1)
            csv_data.loc[sl, 'Arm_Nr'] = arm.arm_num
            csv_data.loc[sl, 'Arm_Richtung'] = arm.direction
            csv_data.loc[sl, 'Arm_Startpreis'] = german_format(arm.start_price)
            csv_data.loc[sl, 'Arm_Endpreis'] = german_format(arm.end_price)
            csv_data.loc[sl, 'Validiert'] = 'Ja' if getattr(arm, "validated", False) else 'Nein'

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir_param, f"HA_Trendarme_{ticker}_{ts}.csv")
    csv_data.to_csv(csv_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"Daten und Trendarme in {csv_path} gespeichert.")
    return csv_path


def save_ha_kerzen_csv(ha_data: pd.DataFrame, output_dir_param: str, ticker: str) -> str:
    os.makedirs(output_dir_param, exist_ok=True)

    def german_format(x):
        try:
            return f"{float(x):,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (ValueError, TypeError):
            return "" if (x is None or str(x).strip() == "") else "0,000"

    if 'Zeit' not in ha_data.columns:
        raise ValueError("Die Spalte 'Zeit' muss im ha_data DataFrame vorhanden sein, um die CSV korrekt zu formatieren.")
    ha_data = ha_data.copy()
    ha_data['Zeit'] = pd.to_datetime(ha_data['Zeit'])

    csv_data = pd.DataFrame({
        'Kerze_Nr': ha_data.index.values,
        'Zeit': ha_data['Zeit'].dt.strftime('%d.%m.%Y %H:%M'),
        'Open': ha_data['Open'].apply(german_format),
        'High': ha_data['High'].apply(german_format),
        'Low': ha_data['Low'].apply(german_format),
        'Close': ha_data['Close'].apply(german_format),
    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir_param, f"HA_Kerzen_{ticker}_{ts}.csv")
    csv_data.to_csv(csv_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"HA-Kerzen wurden in {csv_path} gespeichert.")
    return csv_path


def dump_plot_arms_to_txt(plot_arms: List[ArmConnection], file_path: str = "output/plot_arms_debug.txt"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"\n[INPUT] Validated Arms ({datetime.now().isoformat()}):\n")
        for i, arm in enumerate(plot_arms):
            f.write(f"  C{i+1}: {arm.start_idx}-{arm.end_idx}, Richtung: {arm.direction}, validated: {arm.validated}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Anzahl Plot-Arms: {len(plot_arms)}\n")

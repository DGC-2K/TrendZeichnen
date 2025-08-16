# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import os
import threading
import pandas.api.types as pd_types
import datetime

class ArmConnection:
    def __init__(self, arm_num: int, direction: str, start_idx: int, end_idx: int,
                 start_price: float, end_price: float, validated: bool = False):
        self._arm_num = arm_num
        self._direction = direction
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._start_price = start_price
        self._end_price = end_price
        self._validated = validated
        self._color = 'blue' if validated else 'red'
        self._pt_price: float = np.inf
        self._ph_price: float = -np.inf
        self._ph_candle_idx: int = -1
        self._pt_candle_idx: int = -1
        self._lock = threading.Lock()
        self._connection_type: str = "UNDEFINED"
        self._bounds_source_arm_num: int = -1
        self._retracement_38_2_passed: bool = False
        self._broken = False

    @property
    def broken(self) -> bool:
        return self._broken

    @broken.setter
    def broken(self, value: bool) -> None:
        self._broken = value

    @property
    def arm_num(self) -> int:
        return self._arm_num

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def start_idx(self) -> int:
        return self._start_idx

    @property
    def end_idx(self) -> int:
        return self._end_idx

    @property
    def start_price(self) -> float:
        return self._start_price

    @property
    def end_price(self) -> float:
        return self._end_price

    @property
    def validated(self) -> bool:
        return self._validated

    @validated.setter
    def validated(self, value: bool) -> None:
        with self._lock:
            self._validated = value
            self._color = 'blue' if value else 'red'

    @property
    def color(self) -> str:
        if self._broken:
            return 'gray'
        return self._color

    @property
    def pt_price(self) -> float:
        return self._pt_price

    @property
    def ph_price(self) -> float:
        return self._ph_price

    @property
    def ph_candle_idx(self) -> int:
        return self._ph_candle_idx

    @property
    def pt_candle_idx(self) -> int:
        return self._pt_candle_idx

    @property
    def bounds_source_arm_num(self) -> int:
        return self._bounds_source_arm_num

    @bounds_source_arm_num.setter
    def bounds_source_arm_num(self, value: int) -> None:
        with self._lock:
            self._bounds_source_arm_num = value

    @property
    def connection_type(self) -> str:
        return self._connection_type

    @property
    def retracement_38_2_passed(self) -> bool:
        return self._retracement_38_2_passed

    def update_pt(self, price: float) -> None:
        with self._lock:
            self._pt_price = min(self._pt_price, price)

    def update_ph(self, price: float) -> None:
        with self._lock:
            self._ph_price = max(self._ph_price, price)

    def set_ph_data(self, price: float, idx: int) -> None:
        with self._lock:
            if price > self._ph_price:
                self._ph_price = price
                self._ph_candle_idx = idx

    def set_pt_data(self, price: float, idx: int) -> None:
        with self._lock:
            if price < self._pt_price:
                self._pt_price = price
                self._pt_candle_idx = idx

    def reset_connection_type(self, ha_data: pd.DataFrame) -> None:
        with self._lock:
            self._connection_type = "UNDEFINED"

    def check_retracement(self, ha_data: pd.DataFrame, next_arm_start_idx: int) -> None:
        try:
            arm_data = ha_data.iloc[self.start_idx:self.end_idx + 1]
            if arm_data.empty:
                return

            arm_high = float(arm_data['High'].max())
            arm_low = float(arm_data['Low'].min())
            correction_data = ha_data.iloc[self.end_idx + 1:next_arm_start_idx]

            if self.direction == 'UP':
                total_movement = arm_high - arm_low
                retracement_level = arm_high - (total_movement * 0.382)
                if not correction_data.empty and correction_data['Low'].min() < retracement_level:
                    self._retracement_38_2_passed = True
            elif self.direction == 'DOWN':
                total_movement = arm_high - arm_low
                retracement_level = arm_low + (total_movement * 0.382)
                if not correction_data.empty and correction_data['High'].max() > retracement_level:
                    self._retracement_38_2_passed = True
        except (KeyError, IndexError, ValueError):
            self._retracement_38_2_passed = False


class ArmBoundaries:
    def __init__(self, upper: float, lower: float):
        self._upper = upper
        self._lower = lower
        self._lock = threading.Lock()

    @property
    def upper(self) -> float:
        with self._lock:
            return self._upper

    @upper.setter
    def upper(self, value: float) -> None:
        with self._lock:
            self._upper = value

    @property
    def lower(self) -> float:
        with self._lock:
            return self._lower

    @lower.setter
    def lower(self, value: float) -> None:
        with self._lock:
            self._lower = value

    def contains_ph(self, price: float) -> bool:
        with self._lock:
            return price > self._upper

    def contains_pt(self, price: float) -> bool:
        with self._lock:
            return price < self._lower

    def __str__(self):
        return f"Upper: {self.upper:.2f}, Lower: {self.lower:.2f}"


class ArmContainer:
    def __init__(self, debug_mode: bool = True):
        self.arms: List[ArmConnection] = []
        self.broken_arms: List[ArmConnection] = []
        self.bounds = ArmBoundaries(upper=np.inf, lower=-np.inf)
        self.bounds_source_arm_num: int = 0
        self._lock = threading.Lock()
        self.debug_mode = debug_mode
        self.plot_arms: List[ArmConnection] = []

    def add_arm(self, arm: ArmConnection) -> None:
        with self._lock:
            self.arms.append(arm)

    def update_plot_arms(self, validated_arms: list, ha_data):
        verbindungen_liste = berechne_verbindungslinien(validated_arms, ha_data)
        self.plot_arms = []
        debug_path = r"D:\TradingBot\output\C-Serie-Debug-Ausgaben4.txt"

        with open(debug_path, "a", encoding="utf-8") as dbgfile:
            dbgfile.write(f"\n[INPUT] Validated Arms für update_plot_arms ({datetime.datetime.now().isoformat()}):\n")
            for i, v in enumerate(validated_arms):
                dbgfile.write(f"  B{i+1}: {v.start_idx}-{v.end_idx}, Richtung: {v.direction}, validated: {v.validated}\n")
            dbgfile.write("-" * 50 + "\n")

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

            if 'mitte' in verbindung:
                mid_idx, mid_price = verbindung['mitte']
                if direction == 'UP':
                    arm.set_pt_data(mid_price, mid_idx)
                else:
                    arm.set_ph_data(mid_price, mid_idx)

            self.plot_arms.append(arm)

        with open(debug_path, "a", encoding="utf-8") as dbgfile:
            dbgfile.write(f"\n[OUTPUT] Plot Arms nach update_plot_arms ({datetime.datetime.now().isoformat()}):\n")
            for i, arm in enumerate(self.plot_arms):
                dbgfile.write(
                    f"  C{i+1}: Kerzen {arm.start_idx}-{arm.end_idx}, Richtung: {arm.direction}, "
                    f"StartPreis: {arm.start_price}, EndPreis: {arm.end_price}, validated: {arm.validated}"
                )
                if hasattr(arm, "pt_idx") and arm.pt_idx is not None:
                    dbgfile.write(f", PT: {arm.pt_idx} ({arm.pt if hasattr(arm, 'pt') else ''})")
                if hasattr(arm, "ph_idx") and arm.ph_idx is not None:
                    dbgfile.write(f", PH: {arm.ph_idx} ({arm.ph if hasattr(arm, 'ph') else ''})")
                dbgfile.write("\n")
            dbgfile.write("-" * 50 + "\n")

    def validate_arms(self, ha_data: pd.DataFrame) -> None:
        if ha_data.empty or not self.arms:
            print("?? Warnung: Keine Daten oder Arme vorhanden.")
            return

        with self._lock:
            first_arm = self.arms[0]
            first_arm.validated = True
            first_arm.validation_candles = None

            if first_arm.end_idx < len(ha_data) and first_arm.start_idx < len(ha_data):
                arm_data = ha_data.iloc[first_arm.start_idx:first_arm.end_idx + 1]
                self.bounds.upper = float(arm_data['High'].max())
                self.bounds.lower = float(arm_data['Low'].min())
                self.bounds_source_arm_num = first_arm.arm_num
                first_arm.bounds_source_arm_num = first_arm.arm_num
            else:
                self.bounds = ArmBoundaries(upper=np.inf, lower=-np.inf)

            for i in range(1, len(self.arms)):
                arm = self.arms[i]
                arm.validated = False
                arm.validation_candles = None
                arm.bounds_source_arm_num = self.bounds_source_arm_num

                ref_upper = self.bounds.upper
                ref_lower = self.bounds.lower

                v_start = arm.start_idx
                v_end = self.arms[i + 1].end_idx if (i + 1 < len(self.arms)) else len(ha_data) - 1

                valid_kette = None
                for j in range(v_start, v_end - 1):
                    idxs = [j, j + 1, j + 2]
                    if idxs[-1] > v_end:
                        break
                    closes = [float(ha_data.iloc[k]['Close']) for k in idxs]
                    if all(c > ref_upper for c in closes) or all(c < ref_lower for c in closes):
                        if any(arm.start_idx <= k <= arm.end_idx for k in idxs):
                            valid_kette = idxs
                            break

                if valid_kette:
                    arm.validated = True
                    arm.validation_candles = valid_kette

                if arm.validated:
                    self._update_bounds(arm, ha_data)
                    naechste_kerze_idx = arm.end_idx + 1
                    self._adjust_bounds_to_candle_extremes(ha_data, naechste_kerze_idx, None)

            self.update_plot_arms([arm for arm in self.arms if arm.validated], ha_data)
            dump_plot_arms_to_txt(self.plot_arms, prefix="Plot-Arms-Dump", filename=r"D:\TradingBot\output\DumpPA.txt")
            from trend_break_detector import TrendBreakDetector

            dump_plot_arms_to_txt(self.plot_arms, prefix="Vor Trendbruch-Check", filename=r"D:\TradingBot\output\DumpPA2.txt")
            TrendBreakDetector.detect_trend_break_and_restart(self, ha_data)
            dump_plot_arms_to_txt(self.plot_arms, prefix="Nach Trendbruch-Check", filename=r"D:\TradingBot\output\DumpPA2.txt")

    def _adjust_bounds_to_candle_extremes(self, ha_data: pd.DataFrame, target_idx: int, debug_file) -> None:
        if target_idx >= len(ha_data):
            return

        try:
            candle = ha_data.iloc[target_idx]
            candle_high = float(candle['High'])
            candle_low = float(candle['Low'])
            candle_open = float(candle['Open'])
            candle_close = float(candle['Close'])

            candle_body_min = min(candle_open, candle_close)
            candle_body_max = max(candle_open, candle_close)

            if self.bounds.upper >= candle_body_max and self.bounds.upper <= candle_high:
                self.bounds.upper = candle_high

            if self.bounds.lower <= candle_body_min and self.bounds.lower >= candle_low:
                self.bounds.lower = candle_low

        except (KeyError, ValueError, IndexError):
            pass

    def _update_bounds(self, arm: ArmConnection, data: pd.DataFrame) -> None:
        if arm.end_idx >= len(data) or arm.start_idx >= len(data):
            print(f"?? Warnung: _update_bounds erhielt Arm {arm.arm_num} mit ungültigen Indizes. Bounds nicht aktualisiert.")
            return

        arm_data = data.iloc[arm.start_idx:arm.end_idx + 1]
        self.bounds.upper = float(arm_data['High'].max())
        self.bounds.lower = float(arm_data['Low'].min())
        self.bounds_source_arm_num = arm.arm_num


def calculate_ha(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        print("?? Warnung: calculate_ha wurde mit leeren Daten aufgerufen.")
        return pd.DataFrame()

    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in data.columns:
            raise KeyError(f"Fehlende Spalte: '{col}'")
        if not pd_types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data.dropna(subset=[col], inplace=True)
            if data.empty:
                raise ValueError(f"Daten nach Konvertierung leer für Spalte: '{col}'")

    open_p = data['Open'].values
    high_p = data['High'].values
    low_p = data['Low'].values
    close_p = data['Close'].values

    ha_close = (open_p + high_p + low_p + close_p) / 4
    ha_open = np.zeros_like(ha_close)

    if len(ha_open) > 0:
        ha_open[0] = (open_p[0] + close_p[0]) / 2

    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    ha_high = np.maximum(high_p, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low_p, np.minimum(ha_open, ha_close))

    trend = np.where(ha_close > ha_open, 'UP',
                    np.where(ha_close < ha_open, 'DOWN', 'FLAT'))

    for i in range(len(trend)):
        if trend[i] == 'FLAT':
            for j in range(i + 1, len(trend)):
                if trend[j] in ('UP', 'DOWN'):
                    trend[i] = trend[j]
                    break
            else:
                for j in range(i - 1, -1, -1):
                    if trend[j] in ('UP', 'DOWN'):
                        trend[i] = trend[j]
                        break
                else:
                    trend[i] = 'UP'

    result_df = pd.DataFrame({
        'Open': ha_open,
        'High': ha_high,
        'Low': ha_low,
        'Close': ha_close,
        'Trend': trend
    }, index=data.index)

    if 'Zeit' in data.columns:
        result_df['Zeit'] = data['Zeit']

    return result_df


def kerzenfarbe(open_, close_):
    if close_ > open_:
        return "bullish"
    elif close_ < open_:
        return "bearish"
    else:
        return "doji"

def remove_micro_flip_candles(
    df: pd.DataFrame,
    body_ratio: float = 0.20,
    range_ratio: float = 0.50,
    require_inside: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    # --- harte Typen: Float erzwingen ---
    for c in ("Open","High","Low","Close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"]).reset_index(drop=True)

    # Farbe sicher bestimmen (Fallback ohne 'Trend')
    def color_at(i: int) -> str:
        r = df.iloc[i]
        if "Trend" in df.columns and r["Trend"] in ("UP","DOWN"):
            return r["Trend"]
        return "UP" if float(r["Close"]) >= float(r["Open"]) else "DOWN"

    n = len(df)
    if n < 3:
        return df

    to_drop = []
    for i in range(1, n-1):
        t_prev, t_curr, t_next = color_at(i-1), color_at(i), color_at(i+1)
        # isolierter Farbflip (… und Nachbarn gleiche Farbe)
        if (t_curr != t_prev) and (t_prev == t_next):
            b_prev = abs(df.iloc[i-1]["Close"] - df.iloc[i-1]["Open"])
            b_curr = abs(df.iloc[i]["Close"]  - df.iloc[i]["Open"])
            b_next = abs(df.iloc[i+1]["Close"] - df.iloc[i+1]["Open"])

            r_prev = df.iloc[i-1]["High"] - df.iloc[i-1]["Low"]
            r_curr = df.iloc[i]["High"]     - df.iloc[i]["Low"]
            r_next = df.iloc[i+1]["High"] - df.iloc[i+1]["Low"]

            b_avg = (b_prev + b_next) / 2.0
            r_avg = (r_prev + r_next) / 2.0

            small_body = (b_avg > 0) and (b_curr <= body_ratio  * b_avg)
            small_range = (r_avg > 0) and (r_curr <= range_ratio * r_avg)

            inside_ok = True
            if require_inside:
                hi_nei = max(df.iloc[i-1]["High"], df.iloc[i+1]["High"])
                lo_nei = min(df.iloc[i-1]["Low"],  df.iloc[i+1]["Low"])
                inside_ok = (df.iloc[i]["High"] <= hi_nei) and (df.iloc[i]["Low"] >= lo_nei)

            if small_body and small_range and inside_ok:
                to_drop.append(i)

    if verbose and to_drop:
        print(f"[cleanup] remove_micro_flip_candles: drop {len(to_drop)} -> {to_drop[:10]}")

    return df.drop(df.index[to_drop]).reset_index(drop=True)


def remove_tiny_same_color_candles(
    df: pd.DataFrame,
    body_ratio: float = 0.08,     # ≤ 8% des Nachbar-Körperdurchschnitts
    range_ratio: float = 0.35,    # ≤ 35% des Nachbar-Range-Durchschnitts
    require_inside: bool = True,  # Kerze muss vollständig im Nachbar-Korridor liegen
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Entfernt winzige HA-Kerzen, die die gleiche Farbe wie BEIDE Nachbarn haben und
    (optional) komplett innerhalb deren High/Low liegen (Inside-Bar).

    Regeln (zusätzlich zum Flip-Filter):
      - Trend[i] == Trend[i-1] == Trend[i+1]
      - |Close[i]-Open[i]| <= body_ratio   * avg(|Close[i±1]-Open[i±1]|)
      - (High[i]-Low[i])   <= range_ratio  * avg((High[i±1]-Low[i±1]))
      - optional: Inside-Bar gegenüber Nachbarn
    """
    # Falls du einen Guard hast:
    try:
        require_ha_mode(df)  # nutzt deine bestehende Guard-Funktion
    except Exception:
        pass

    if len(df) < 3:
        return df

    to_drop = []
    for i in range(1, len(df) - 1):
        t_prev, t_curr, t_next = df.iloc[i-1]["Trend"], df.iloc[i]["Trend"], df.iloc[i+1]["Trend"]
        if t_prev == t_curr == t_next:
            b_prev = abs(df.iloc[i-1]["Close"] - df.iloc[i-1]["Open"])
            b_curr = abs(df.iloc[i]["Close"]  - df.iloc[i]["Open"])
            b_next = abs(df.iloc[i+1]["Close"] - df.iloc[i+1]["Open"])

            r_prev = df.iloc[i-1]["High"] - df.iloc[i-1]["Low"]
            r_curr = df.iloc[i]["High"]  - df.iloc[i]["Low"]
            r_next = df.iloc[i+1]["High"] - df.iloc[i+1]["Low"]

            b_avg = (b_prev + b_next) / 2.0
            r_avg = (r_prev + r_next) / 2.0

            if b_avg > 0 and r_avg > 0 and b_curr <= body_ratio * b_avg and r_curr <= range_ratio * r_avg:
                inside_ok = True
                if require_inside:
                    hi_nei = max(df.iloc[i-1]["High"], df.iloc[i+1]["High"])
                    lo_nei = min(df.iloc[i-1]["Low"],  df.iloc[i+1]["Low"])
                    inside_ok = (df.iloc[i]["High"] <= hi_nei) and (df.iloc[i]["Low"] >= lo_nei)
                if inside_ok:
                    to_drop.append(i)

    if verbose and to_drop:
        print(f"[cleanup] remove_tiny_same_color_candles: drop {len(to_drop)} Kerzen (idx={to_drop[:6]}...)")

    return df.drop(df.index[to_drop]).reset_index(drop=True)


def canonicalize_to_ha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Setzt Open/High/Low/Close auf HA_* und sorgt dafür, dass HA_* existieren.
    Falls HA_* fehlen, werden sie aus den vorhandenen O/H/L/C gespiegelt.
    Sichert bestehende O/H/L/C einmalig in Raw_*.
    """
    df = df.copy()
    ohlc = ("Open", "High", "Low", "Close")
    ha   = ("HA_Open", "HA_High", "HA_Low", "HA_Close")

    # Falls HA_* fehlen: aus O/H/L/C erzeugen
    if not all(col in df.columns for col in ha):
        if not all(col in df.columns for col in ohlc):
            raise ValueError("canonicalize_to_ha: O/H/L/C fehlen – kann HA nicht kanonisieren.")
        df["HA_Open"]  = df["Open"]
        df["HA_High"]  = df["High"]
        df["HA_Low"]   = df["Low"]
        df["HA_Close"] = df["Close"]

    # Raw_* sichern (nur einmal)
    for base in ohlc:
        raw = f"Raw_{base}"
        if base in df.columns and raw not in df.columns:
            df[raw] = df[base]

    # O/H/L/C auf HA_* setzen (ab hier sind O/H/L/C garantiert HA)
    df["Open"]  = df["HA_Open"]
    df["High"]  = df["HA_High"]
    df["Low"]   = df["HA_Low"]
    df["Close"] = df["HA_Close"]

    df.attrs["data_mode"] = "HA"
    return df



def require_ha_mode(df: pd.DataFrame, auto_fix: bool = True) -> None:
    """
    Verlangt HA-only-Mode. Mit auto_fix=True werden fehlende HA_* automatisch
    aus O/H/L/C gespiegelt und O/H/L/C auf HA_* gesetzt.
    Wirft nur dann, wenn weder HA_* noch O/H/L/C vollständig vorhanden sind.
    """
    if df is None or len(df) == 0:
        raise ValueError("require_ha_mode: Leer/None DataFrame.")

    ohlc = ("Open", "High", "Low", "Close")
    ha   = ("HA_Open", "HA_High", "HA_Low", "HA_Close")

    has_ohlc = all(c in df.columns for c in ohlc)
    has_ha   = all(c in df.columns for c in ha)

    if not has_ha:
        if auto_fix and has_ohlc:
            # HA_* aus O/H/L/C erzeugen
            df["HA_Open"]  = df["Open"]
            df["HA_High"]  = df["High"]
            df["HA_Low"]   = df["Low"]
            df["HA_Close"] = df["Close"]
            has_ha = True
        else:
            raise ValueError("require_ha_mode: HA-Spalten fehlen.")

    if not has_ohlc:
        if auto_fix:
            # Falls jemand HA_* geliefert hat aber O/H/L/C fehlen – spiegeln
            df["Open"]  = df["HA_Open"]
            df["High"]  = df["HA_High"]
            df["Low"]   = df["HA_Low"]
            df["Close"] = df["HA_Close"]
            has_ohlc = True
        else:
            raise ValueError("require_ha_mode: O/H/L/C fehlen.")

    # O/H/L/C auf HA_* angleichen (hart erzwingen)
    for base, h in zip(ohlc, ha):
        if base not in df.columns or h not in df.columns:
            raise ValueError("require_ha_mode: Spalteninkonsistenz.")
        if not df[base].equals(df[h]):
            df[base] = df[h]

    df.attrs["data_mode"] = "HA"



def remove_isolated_candles(df: pd.DataFrame) -> pd.DataFrame:
    # Guards
    if df is None or len(df) < 3:
        return df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else df

    # Primär HA – Fallback auf klassische OHLC, mit einmaligem Hinweis
    if {"HA_Open", "HA_Close"}.issubset(df.columns):
        o_col, c_col = "HA_Open", "HA_Close"
    elif {"Open", "Close"}.issubset(df.columns):
        o_col, c_col = "Open", "Close"
        print("?? remove_isolated_candles: HA_Open/HA_Close nicht gefunden – Fallback auf Open/Close.")
    else:
        raise ValueError("remove_isolated_candles benötigt entweder HA_Open/HA_Close oder Open/Close.")

    to_remove = []
    for i in range(1, len(df) - 1):
        prev_color = kerzenfarbe(df.iloc[i - 1][o_col], df.iloc[i - 1][c_col])
        curr_color = kerzenfarbe(df.iloc[i][o_col], df.iloc[i][c_col])
        next_color = kerzenfarbe(df.iloc[i + 1][o_col], df.iloc[i + 1][c_col])

        if prev_color == next_color and curr_color != prev_color:
            prev_body = abs(df.iloc[i - 1][o_col] - df.iloc[i - 1][c_col])
            curr_body = abs(df.iloc[i][o_col] - df.iloc[i][c_col])
            next_body = abs(df.iloc[i + 1][o_col] - df.iloc[i + 1][c_col])
            avg_neighbor_body = (prev_body + next_body) / 2.0
            if avg_neighbor_body > 0 and curr_body <= 0.06 * avg_neighbor_body:
                to_remove.append(i)

    if to_remove:
        df = df.drop(df.index[to_remove])
    return df.reset_index(drop=True)


def count_isolated_ha_candles(df: pd.DataFrame) -> int:
    """
    Zählt verbleibende isolierte HA-Kerzen (gleiches Kriterium wie remove_isolated_candles).
    Erwartet 0 nach erfolgreichem Cleanup.
    """
    if df is None or len(df) < 3:
        return 0

    if {"HA_Open", "HA_Close"}.issubset(df.columns):
        o_col, c_col = "HA_Open", "HA_Close"
    elif {"Open", "Close"}.issubset(df.columns):
        o_col, c_col = "Open", "Close"
    else:
        raise ValueError("count_isolated_ha_candles benötigt HA_Open/HA_Close oder Open/Close.")

    cnt = 0
    for i in range(1, len(df) - 1):
        prev_color = kerzenfarbe(df.iloc[i - 1][o_col], df.iloc[i - 1][c_col])
        curr_color = kerzenfarbe(df.iloc[i][o_col], df.iloc[i][c_col])
        next_color = kerzenfarbe(df.iloc[i + 1][o_col], df.iloc[i + 1][c_col])

        if prev_color == next_color and curr_color != prev_color:
            prev_body = abs(df.iloc[i - 1][o_col] - df.iloc[i - 1][c_col])
            curr_body = abs(df.iloc[i][o_col] - df.iloc[i][c_col])
            next_body = abs(df.iloc[i + 1][o_col] - df.iloc[i + 1][c_col])
            avg_neighbor_body = (prev_body + next_body) / 2.0
            if avg_neighbor_body > 0 and curr_body <= 0.06 * avg_neighbor_body:
                cnt += 1
    return cnt


def detect_trend_arms(ha_data: pd.DataFrame) -> List[ArmConnection]:
    if ha_data.empty:
        print("?? Warnung: detect_trend_arms wurde mit leeren HA-Daten aufgerufen.")
        return []

    if len(ha_data) < 1:
        return []

    arms: List[ArmConnection] = []
    current_trend = None
    start_idx = 0
    start_price = 0.0
    extreme_idx = 0
    extreme_price = 0.0
    arm_counter = 1

    for i in range(len(ha_data)):
        candle = ha_data.iloc[i]

        if not all(col in candle for col in ['Trend', 'High', 'Low']):
            print(f"?? Warnung: Fehlende Spalten in HA-Kerze bei Index {i}. Überspringe.")
            continue

        trend = candle['Trend']
        candle_high = float(candle['High'])
        candle_low = float(candle['Low'])

        if current_trend is None and trend in ('UP', 'DOWN'):
            current_trend = trend
            start_idx = i
            start_price = candle_high if trend == 'DOWN' else candle_low
            extreme_idx = i
            extreme_price = candle_low if trend == 'DOWN' else candle_high
            continue

        if current_trend == 'DOWN':
            if trend == 'DOWN':
                if candle_low < extreme_price:
                    extreme_idx = i
                    extreme_price = candle_low
            else:
                arms.append(ArmConnection(
                    arm_counter, current_trend, 
                    start_idx, extreme_idx,
                    start_price, extreme_price
                ))
                arm_counter += 1
                start_idx = extreme_idx
                start_price = extreme_price
                current_trend = 'UP'
                extreme_idx = i
                extreme_price = candle_high

        elif current_trend == 'UP':
            if trend == 'UP':
                if candle_high > extreme_price:
                    extreme_idx = i
                    extreme_price = candle_high
            else:
                arms.append(ArmConnection(
                    arm_counter, current_trend,
                    start_idx, extreme_idx,
                    start_price, extreme_price
                ))
                arm_counter += 1
                start_idx = extreme_idx
                start_price = extreme_price
                current_trend = 'DOWN'
                extreme_idx = i
                extreme_price = candle_low

    if current_trend is not None:
        arms.append(ArmConnection(
            arm_counter, current_trend,
            start_idx, extreme_idx,
            start_price, extreme_price
        ))

    return arms


def find_max_high_in_range(data: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[Optional[float], Optional[int]]:
    search_slice = data.loc[start_idx:end_idx]
    if search_slice.empty:
        return None, None
    max_high = search_slice['High'].max()
    max_high_idx = search_slice['High'].idxmax()
    return max_high, max_high_idx


def find_min_low_in_range(data: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[Optional[float], Optional[int]]:
    search_slice = data.loc[start_idx:end_idx]
    if search_slice.empty:
        return None, None
    min_low = search_slice['Low'].min()
    min_low_idx = search_slice['Low'].idxmin()
    return min_low, min_low_idx


def berechne_verbindungslinien(validated_arms, data):
    verbindungen_liste = []

    # Optional: erste Trendarm-Verbindung (B1) wie gehabt
    if validated_arms:
        arm_0 = validated_arms[0]
        verbindungen_liste.append({
            'start': (arm_0.start_idx, arm_0.start_price),
            'ende': (arm_0.end_idx, arm_0.end_price),
            'typ': 'B1'
        })

    for i in range(len(validated_arms) - 1):
        arm_i = validated_arms[i]
        arm_j = validated_arms[i + 1]

        # A, B, C: jeweilige Extremwerte
        A = arm_i.start_price
        B = arm_i.end_price
        C = arm_j.end_price

        b_idx_pos = arm_i.end_idx
        c_idx_pos = arm_j.end_idx

        # 1) Aufwärtsfall: A < B < C
        if A < B < C:
            # Tiefster Punkt (Low) zwischen B und C (exklusive B und C)
            if b_idx_pos + 1 < c_idx_pos:
                idx_bis_c = data.index[(data.index > b_idx_pos) & (data.index < c_idx_pos)]
                if not idx_bis_c.empty:
                    min_idx = data.loc[idx_bis_c, "Low"].idxmin()
                    D = data.loc[min_idx, "Low"]
                    if D < B:
                        verbindungen_liste.append({
                            'start': (b_idx_pos, B),
                            'mitte': (data.index.get_loc(min_idx), D),
                            'ende': (c_idx_pos, C),
                            'typ': 'B-D-C'
                        })

        # 2) Abwärtsfall: A > B > C
        elif A > B > C:
            # Höchster Punkt (High) zwischen B und C (exklusive B und C)
            if b_idx_pos + 1 < c_idx_pos:
                idx_bis_c = data.index[(data.index > b_idx_pos) & (data.index < c_idx_pos)]
                if not idx_bis_c.empty:
                    max_idx = data.loc[idx_bis_c, "High"].idxmax()
                    D = data.loc[max_idx, "High"]
                    if D > B:
                        verbindungen_liste.append({
                            'start': (b_idx_pos, B),
                            'mitte': (data.index.get_loc(max_idx), D),
                            'ende': (c_idx_pos, C),
                            'typ': 'B-D-C'
                        })

        # 3) C < A < B oder C > A > B
        elif (C < A < B) or (C > A > B):
            verbindungen_liste.append({
                'start': (b_idx_pos, B),
                'ende': (c_idx_pos, C),
                'typ': 'B-C'
            })

    return verbindungen_liste

# --- Dow-Regel: Runs + Annotationen (HA-only) --------------------
from typing import List, Dict, Optional

def build_color_runs(df: pd.DataFrame, trend_col: str = "Trend") -> List[Dict]:
    """
    Segmentiert den DF in 'Runs' aus aufeinanderfolgenden HA-Kerzen gleicher Farbe (UP/DOWN).
    Dojis/Neutrals werden ignoriert bzw. schließen keinen Run auf (werden dem vorherigen zugeschlagen).
    Erwartet: require_ha_mode(df) wurde vorher aufgerufen (HA-only).
    Rückgabe: Liste von Dicts mit start_idx, end_idx, color, run_high, run_low
    """
    require_ha_mode(df)
    if trend_col not in df.columns:
        raise ValueError(f"build_color_runs: Spalte '{trend_col}' fehlt.")

    runs: List[Dict] = []
    n = len(df)
    if n == 0:
        return runs

    def _finalize(s: int, e: int, color: str):
        frag = df.loc[s:e, :]
        runs.append({
            "id": len(runs),
            "start_idx": s,
            "end_idx": e,
            "color": color,
            "run_high": float(frag["High"].max()),
            "run_low": float(frag["Low"].min()),
            # Platzhalter – füllen wir in annotate_dow_per_run
            "regime": "RANGE",
            "hh": False, "hl": False, "lh": False, "ll": False
        })

    i = 0
    # erste gültige Farbe finden
    while i < n and df.at[i, trend_col] not in ("UP", "DOWN"):
        i += 1
    if i >= n:
        return runs

    curr_color = df.at[i, trend_col]
    run_start = i
    i += 1

    while i < n:
        c = df.at[i, trend_col]
        if c not in ("UP", "DOWN"):
            # ignoriere Doji/Neutral innerhalb des Runs
            i += 1
            continue
        if c != curr_color:
            _finalize(run_start, i - 1, curr_color)
            curr_color = c
            run_start = i
        i += 1
    # letzten Run schließen
    _finalize(run_start, n - 1, curr_color)
    return runs


def annotate_dow_per_run(runs: List[Dict]) -> List[Dict]:
    """
    Wendet die Dow-Regel auf Run-Ebene an.
    - UP-Run bestätigt Aufwärtstrend, wenn HH & HL relativ zu zuletzt bestätigten Marken.
    - DOWN-Run bestätigt Abwärtstrend, wenn LL & LH relativ zu zuletzt bestätigten Marken.
    Aktualisiert run['regime'] in {"UP","DOWN","RANGE"} und Flags hh/hl/lh/ll.
    """
    if not runs:
        return runs

    last_high: Optional[float] = None
    last_low: Optional[float] = None
    current_regime: str = "RANGE"

    # Initialisierung mit erstem Run (unbestätigt)
    r0 = runs[0]
    last_high = r0["run_high"]
    last_low = r0["run_low"]
    r0["regime"] = "RANGE"

    for ri in range(1, len(runs)):
        r = runs[ri]
        color = r["color"]
        rh, rl = r["run_high"], r["run_low"]

        # Reset Flags
        r["hh"] = r["hl"] = r["lh"] = r["ll"] = False
        r["regime"] = current_regime  # default: bleibt wie zuvor

        if color == "UP":
            r["hh"] = (rh > (last_high if last_high is not None else rh - 1e9))
            r["hl"] = (rl > (last_low  if last_low  is not None else rl - 1e9))
            if r["hh"] and r["hl"]:
                r["regime"] = "UP"
                current_regime = "UP"
                last_high, last_low = rh, rl   # bestätigte Marken übernehmen
        elif color == "DOWN":
            r["ll"] = (rl < (last_low  if last_low  is not None else rl + 1e9))
            r["lh"] = (rh < (last_high if last_high is not None else rh + 1e9))
            if r["ll"] and r["lh"]:
                r["regime"] = "DOWN"
                current_regime = "DOWN"
                last_high, last_low = rh, rl
        # Falls nur eine der Bedingungen erfüllt ist → unbestätigt: regime bleibt alt

    return runs
def smooth_regimes_by_neighbors(
    arms,
    *,
    lock_conf: float = 0.70,     # ab dieser Confidence bleibt das Mittel unangetastet
    require_same_geom: bool = False  # optional: nur glätten, wenn die drei Arme geometrisch gleich „steigend/fallend“ sind
):
    """
    Wenn links und rechts dasselbe Regime haben und der mittlere Arm abweicht,
    wird der mittlere auf das Nachbar-Regime gesetzt – es sei denn, seine
    eigene Einstufung ist sehr sicher (regime_confidence >= lock_conf).

    Setzt:
      arm.regime, arm.regime_source += " | smoothed:neighbors"
    Gibt die Anzahl der Anpassungen zurück.
    """
    def _arm_dir(a: "ArmConnection") -> str:
        sp = float(getattr(a, "start_price", 0.0))
        ep = float(getattr(a, "end_price",   0.0))
        return "UP" if ep >= sp else "DOWN"

    changes = 0
    for i in range(1, len(arms) - 1):
        left, mid, right = arms[i - 1], arms[i], arms[i + 1]
        lreg = getattr(left,  "regime", None)
        mreg = getattr(mid,   "regime", None)
        rreg = getattr(right, "regime", None)

        if lreg in ("UP","DOWN") and rreg in ("UP","DOWN") and lreg == rreg:
            if mreg not in ("UP","DOWN") or mreg != lreg:
                conf = float(getattr(mid, "regime_confidence", 0.0) or 0.0)
                if conf < lock_conf:
                    if not require_same_geom or (_arm_dir(left) == _arm_dir(mid) == _arm_dir(right)):
                        mid.regime = lreg
                        src = getattr(mid, "regime_source", "unknown")
                        mid.regime_source = f"{src} | smoothed:neighbors"
                        changes += 1
    return changes

def annotate_arms_with_runs(
    arms: List["ArmConnection"],
    runs: List[dict],
    ha_data: Optional[pd.DataFrame] = None,
    *,
    geometry_guard_factor: float = 1.25,   # Arm-Delta >= 1.25 * Median-Range -> Geometrie erzwingen
    neighbor_align_factor: float = 0.60,   # 1-3-Glättung: mittlere Amplitude < 60% der Nachbarn -> angleichen
    do_neighbor_smoothing: bool = True,
    use_h1: bool = True,                   # H1 (HH/HL vs. LL/LH) aktivieren
    h1_min_events: int = 2,
    h1_dominance: float = 0.60,
    debug: bool = False,
    debug_file: Optional[str] = None,
) -> List["ArmConnection"]:
    """
    Weist jedem Arm ein Regime ('UP'/'DOWN') zu.

    Reihenfolge/Logik pro Arm:
      1) Falls möglich H1 (Pivots, HH/HL vs. LL/LH) -> hat Vorrang.
      2) Sonst Run-Mehrheit: Gewichte = Summe(High-Low) in überlappendem Fenster.
      3) Geometrie-Guard: Wenn Mehrheit != Geometrie und |Δ| groß genug,
         setze auf Geometrie.
      4) Optional Nachbar-Glättung (1-3) und harte Block-Kohärenz.

    Setzt:
      arm.regime -> 'UP'/'DOWN'
      arm.regime_source -> 'H1' | 'runs-majority' | 'geom-override' (+ evtl. Glättungszusätze)
      arm.regime_confidence -> [0..1]
    """

    if not arms or not runs:
        return arms

    # ---------- Hilfen ----------
    def _clip_idx(i: int, n: int) -> int:
        return max(0, min(n - 1, int(i)))

    def _overlap(a0: int, a1: int, b0: int, b1: int) -> Tuple[Optional[int], Optional[int]]:
        s = max(a0, b0)
        e = min(a1, b1)
        return (s, e) if s <= e else (None, None)

    def _run_bounds(run: dict) -> Tuple[int, int]:
        s = run.get("start_idx", run.get("start", run.get("s")))
        e = run.get("end_idx",   run.get("ende",  run.get("end", run.get("e"))))
        if isinstance(s, (tuple, list)): s = s[0]
        if isinstance(e, (tuple, list)): e = e[0]
        return int(s), int(e)

    def _run_regime(run: dict) -> Optional[str]:
        r = run.get("regime")
        if r not in ("UP", "DOWN"):
            r = run.get("trend", run.get("Trend", run.get("color")))
        if not r:
            return None
        r = str(r).upper()
        if r in ("UP", "DOWN"):
            return r
        if r in ("GREEN", "GRUEN", "GRÜN", "BULL", "BULLISH", "+", "POS"):
            return "UP"
        if r in ("RED", "BEAR", "BEARISH", "-", "NEG"):
            return "DOWN"
        return None

    def _geom_dir(a: "ArmConnection") -> str:
        sp = float(getattr(a, "start_price", 0.0))
        ep = float(getattr(a, "end_price",   0.0))
        return "UP" if ep > sp else "DOWN"

    # ---------- Vorrechnungen für Gewichte / H1 ----------
    n_rows = len(ha_data) if isinstance(ha_data, pd.DataFrame) else 0
    use_weights = bool(n_rows and {"High", "Low"}.issubset(ha_data.columns))
    hl_all = (ha_data["High"].astype(float) - ha_data["Low"].astype(float)).to_numpy() if use_weights else None

    pivots = []
    if use_h1 and isinstance(ha_data, pd.DataFrame) and {"High", "Low"}.issubset(ha_data.columns):
        try:
            pivots = compute_pivots(ha_data, win=2)
        except Exception:
            pivots = []

    def _h1_for_arm(a: "ArmConnection") -> Tuple[Optional[str], float, dict]:
        """Gibt (regime, confidence, details) zurück."""
        if not pivots:
            return None, 0.0, {}
        i0, i1 = int(a.start_idx), int(a.end_idx)
        seg = [p for p in pivots if i0 <= p["idx"] <= i1]
        if len(seg) < 2:
            return None, 0.0, {}

        hh = hl = lh = ll = 0
        last_H = None
        last_L = None
        for p in seg:
            if p["type"] == "H":
                if last_H is not None:
                    if p["price"] > last_H: hh += 1
                    else:                   lh += 1
                last_H = p["price"]
            else:
                if last_L is not None:
                    if p["price"] < last_L: ll += 1
                    else:                   hl += 1
                last_L = p["price"]

        up_cnt = hh + hl
        dn_cnt = ll + lh
        total = up_cnt + dn_cnt
        if total >= h1_min_events:
            if up_cnt >= h1_dominance * total and up_cnt > dn_cnt:
                conf = up_cnt / total
                return "UP", conf, {"hh": hh, "hl": hl, "lh": lh, "ll": ll, "total": total}
            if dn_cnt >= h1_dominance * total and dn_cnt > up_cnt:
                conf = dn_cnt / total
                return "DOWN", conf, {"hh": hh, "hl": hl, "lh": lh, "ll": ll, "total": total}
        return None, 0.0, {"hh": hh, "hl": hl, "lh": lh, "ll": ll, "total": total}

    dbg_lines = []

    # ---------- Hauptschleife über Arme ----------
    for idx, arm in enumerate(arms, 1):
        s = int(getattr(arm, "start_idx"))
        e = int(getattr(arm, "end_idx"))
        if s > e:
            s, e = e, s

        # 1) H1 (falls möglich)
        h1_regime, h1_conf, h1_det = _h1_for_arm(arm)

        # 2) Run-Mehrheit (Gewichte)
        up_w = 0.0
        dn_w = 0.0
        for run in runs:
            rr = _run_regime(run)
            if rr not in ("UP", "DOWN"):
                continue
            rs, re = _run_bounds(run)
            os, oe = _overlap(s, e, rs, re)
            if os is None:
                continue

            if use_weights:
                cs = _clip_idx(os, n_rows); ce = _clip_idx(oe, n_rows)
                w = float(np.nansum(hl_all[cs:ce+1])) if ce >= cs else 0.0
            else:
                w = float(oe - os + 1)

            if rr == "UP": up_w += w
            else:          dn_w += w

        # Mehrheit + Confidence (0..1)
        if up_w == dn_w == 0.0:
            majority = None
            maj_conf = 0.0
        else:
            majority = "UP" if up_w > dn_w else "DOWN"
            maj_conf = abs(up_w - dn_w) / max(up_w + dn_w, 1e-12)

        # 3) Geometrie-Guard
        geom = _geom_dir(arm)
        delta = float(getattr(arm, "end_price") - getattr(arm, "start_price"))
        if use_weights:
            cs = _clip_idx(s, n_rows); ce = _clip_idx(e, n_rows)
            med_rng = float(np.nanmedian(hl_all[cs:ce+1])) if ce >= cs else 0.0
        else:
            med_rng = 0.0

        # 4) Finale Entscheidung, Priorität: H1 -> Mehrheit -> Geometrie
        if h1_regime in ("UP", "DOWN"):
            final = h1_regime
            src = "H1"
            conf = max(h1_conf, maj_conf)  # konservativ: best available
        elif majority in ("UP", "DOWN"):
            if majority != geom and med_rng > 0.0 and abs(delta) >= geometry_guard_factor * med_rng:
                final = geom
                src = "geom-override"
                conf = 1.0
            else:
                final = majority
                src = "runs-majority"
                conf = maj_conf
        else:
            final = geom
            src = "geometry"
            conf = 0.5

        # Setzen
        setattr(arm, "regime", final)
        setattr(arm, "regime_source", src)
        setattr(arm, "regime_confidence", float(conf))

        if debug:
            dbg_lines.append(
                f"C{idx}: [{s}-{e}] geom={geom} Δ={delta:.6f} "
                f"H1={h1_regime} (conf={h1_conf:.2f}, det={h1_det}) "
                f"runs(up={up_w:.3f}, down={dn_w:.3f}, maj={majority}, maj_conf={maj_conf:.2f}) "
                f"med_rng={med_rng:.6f} -> final={final} ({src})"
            )

    # ---------- Nachbar-Glättungen ----------
    if do_neighbor_smoothing and len(arms) >= 3:
        try:
            # weiche 1-3-Glättung (Confidence-basiert)
            smooth_regimes_by_neighbors(arms, lock_conf=0.80, require_same_geom=False)
        except Exception:
            pass
        try:
            # harte Block-Kohärenz
            enforce_block_coherence(arms)
        except Exception:
            pass

    # ---------- Debug-Output ----------
    if debug and dbg_lines:
        if debug_file:
            try:
                with open(debug_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(dbg_lines) + "\n")
            except Exception:
                print("\n".join(dbg_lines))
        else:
            print("\n".join(dbg_lines))

    return arms



def smooth_regimes_by_neighbors(arms, *, lock_conf: float = 0.70, require_same_geom: bool = False):
    """
    Soft-Glättung: Wenn Nachbar-Regime (links/rechts) gleich sind und der mittlere Arm
    abweicht, setze ihn auf das Nachbar-Regime — außer seine eigene
    regime_confidence ist >= lock_conf. Optional kann require_same_geom erzwingen,
    dass auch die geometrische Richtung (start/end) übereinstimmen muss.
    """
    def _geom_dir(a):
        sp = float(getattr(a, "start_price", 0.0))
        ep = float(getattr(a, "end_price",   0.0))
        return "UP" if ep >= sp else "DOWN"

    changes = 0
    for i in range(1, len(arms) - 1):
        left, mid, right = arms[i - 1], arms[i], arms[i + 1]
        lreg = getattr(left,  "regime", None)
        mreg = getattr(mid,   "regime", None)
        rreg = getattr(right, "regime", None)
        if lreg in ("UP","DOWN") and rreg in ("UP","DOWN") and lreg == rreg:
            conf = float(getattr(mid, "regime_confidence", 0.0) or 0.0)
            if conf < lock_conf:
                if not require_same_geom or (_geom_dir(left) == _geom_dir(mid) == _geom_dir(right)):
                    if mreg != lreg:
                        mid.regime = lreg
                        src = getattr(mid, "regime_source", "unknown")
                        mid.regime_source = f"{src} | smoothed:neighbors"
                        changes += 1
    return changes


def enforce_block_coherence(arms):
    """
    Harte Block-Glättung: Wenn Nachbar-Regime (links/rechts) gleich sind,
    setze den mittleren Arm IMMER auf dieses Regime — ohne Confidence-Bedingung.
    Nutze diese Funktion NACH annotate_arms_with_runs und ggf. nach
    smooth_regimes_by_neighbors.
    """
    changes = 0
    for i in range(1, len(arms) - 1):
        left, mid, right = arms[i - 1], arms[i], arms[i + 1]
        lreg = getattr(left,  "regime", None)
        rreg = getattr(right, "regime", None)
        if lreg in ("UP","DOWN") and rreg == lreg:
            if getattr(mid, "regime", None) != lreg:
                mid.regime = lreg
                src = getattr(mid, "regime_source", "unknown")
                mid.regime_source = f"{src} | enforced:block"
                changes += 1
    return changes

# --- NEW: Dow/H1-Tools -------------------------------------------------

def compute_pivots(df: pd.DataFrame, win: int = 2) -> List[dict]:
    """
    Markiert Pivot-Highs/-Lows mit einfacher Fractal-Logik.
    win=2 => Vergleich mit 2 linken und 2 rechten Nachbarn.
    Rückgabe: Liste von dicts: {"idx": i, "type": "H"|"L", "price": float}
    """
    H = df["High"].values
    L = df["Low"].values
    n = len(df)
    pivots = []
    for i in range(win, n - win):
        hi_block = H[i-win:i+win+1]
        lo_block = L[i-win:i+win+1]
        if H[i] == hi_block.max() and (hi_block.argmax() == win):
            pivots.append({"idx": i, "type": "H", "price": float(H[i])})
        if L[i] == lo_block.min() and (lo_block.argmin() == win):
            pivots.append({"idx": i, "type": "L", "price": float(L[i])})
    pivots.sort(key=lambda p: p["idx"])
    return pivots


def arm_regime_by_H1(arm, pivots: List[dict],
                     min_events: int = 2, dominance: float = 0.60) -> Optional[str]:
    """
    Bestimmt das Regime eines Arms nach Dow/H1.
    - Zählt HH/HL und LL/LH nur mit Pivots, die innerhalb [start_idx, end_idx] liegen.
    - Liefert 'UP'/'DOWN' wenn (HH+HL) bzw. (LL+LH) >= dominance * total
      und total >= min_events. Sonst None.
    """
    i0, i1 = int(arm.start_idx), int(arm.end_idx)
    seg = [p for p in pivots if i0 <= p["idx"] <= i1]
    if len(seg) < 2:
        return None

    hh = hl = lh = ll = 0
    last_H = None
    last_L = None
    for p in seg:
        if p["type"] == "H":
            if last_H is not None:
                if p["price"] > last_H:
                    hh += 1
                else:
                    lh += 1
            last_H = p["price"]
        else:  # "L"
            if last_L is not None:
                if p["price"] < last_L:
                    ll += 1
                else:
                    hl += 1
            last_L = p["price"]

    up = hh + hl
    down = ll + lh
    total = up + down
    if total >= min_events:
        if up >= dominance * total and up > down:
            return "UP"
        if down >= dominance * total and down > up:
            return "DOWN"
    return None


# -----------------------------------------------------------------


def debug_luecken_untersuchung(
    arm_i, arm_j, data, d_idx=None, D=None, debug_path=r"D:\TradingBot\output\Luecken_Untersuchung.txt"
):
    import datetime

    arm_i_repr = getattr(arm_i, "_debug_idx", "?")
    arm_j_repr = getattr(arm_j, "_debug_idx", "?")
    zeit_B = data.loc[arm_i.end_idx, 'Zeit'] if hasattr(data, "loc") and arm_i.end_idx in data.index else "?"
    zeit_C = data.loc[arm_j.end_idx, 'Zeit'] if hasattr(data, "loc") and arm_j.end_idx in data.index else "?"

    # Zeilen nach deinem Wunsch
    with open(debug_path, "a", encoding="utf-8") as dbgfile:
        current_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
        dbgfile.write(f"\n[LUECKE-DEBUG] Start {current_time}\n")
        dbgfile.write(f"arm_i -> validated_arms[{arm_i_repr}]\n")
        dbgfile.write(f"arm_j -> validated_arms[{arm_j_repr}]\n")
        dbgfile.write(f"arm_i.direction -> {arm_i.direction}\n")
        dbgfile.write(f"arm_j.direction -> {arm_j.direction}\n")
        dbgfile.write(f"A -> {arm_i.start_price} = arm_i.start_idx -> {arm_i.start_idx}\n")
        dbgfile.write(f"B -> {arm_i.end_price} = arm_i.end_idx -> {arm_i.end_idx}\n")
        dbgfile.write(f"C -> {arm_j.end_price} = arm_j.end_idx -> {arm_j.end_idx}\n")
        if D is not None and d_idx is not None:
            dbgfile.write(f"D -> {D} = d_idx -> {d_idx}\n")
        elif D is not None:
            dbgfile.write(f"D -> {D}\n")
        else:
            dbgfile.write(f"D -> ?????\n")
        dbgfile.write(f"\nzeit_B -> {zeit_B}\n")
        dbgfile.write(f"zeit_C -> {zeit_C}\n")


def dump_plot_arms_to_txt(plot_arms, prefix="Plot-Arms-Dump", filename=r"D:\TradingBot\output\DumpPA.txt"):
    import datetime
    dt = datetime.datetime.now()
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n[{prefix}] ({dt.strftime('%Y-%m-%d %H:%M:%S.%f')}):\n")
        for idx, arm in enumerate(plot_arms, start=1):
            f.write(
                f"  C{idx}: start_idx={getattr(arm, 'start_idx', '-')}, end_idx={getattr(arm, 'end_idx', '-')}, Richtung: {getattr(arm, 'direction', '-')}, broken: {getattr(arm, 'broken', '-')}, validated: {getattr(arm, 'validated', '-')}\n"
            )
        f.write("-" * 50 + "\n")
        f.write(f"Anzahl Plot-Arms: {len(plot_arms)}\n")

        f.write("-" * 50 + "\n")
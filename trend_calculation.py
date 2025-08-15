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
def compute_fib382_for_arms(df: pd.DataFrame, arms: list):
    """
    Berechnet für jeden Arm das 38,2%-Retracement-Level bezogen auf die Arm-Spanne.
    - Erwartet HA-Mode (require_ha_mode)
    - Schreibt arm.fib382 (float) oder None
    - Gibt eine Liste einfacher Dicts (arm_idx, start_idx, end_idx, level) zurück
    """
    require_ha_mode(df)

    out = []
    for idx, arm in enumerate(arms):
        # kompatibel zu deiner ArmConnection: start_idx, extreme_idx, start_price, extreme_price, trend ('UP'/'DOWN')
        s_idx = getattr(arm, "start_idx", None)
        e_idx = getattr(arm, "end_idx", getattr(arm, "extreme_idx", None))
        sp    = float(getattr(arm, "start_price", float("nan")))
        ep    = float(getattr(arm, "extreme_price", float("nan")))
        trend = getattr(arm, "trend", getattr(arm, "current_trend", None))

        if s_idx is None or e_idx is None or e_idx <= s_idx:
            level = None
        else:
            low, high = (min(sp, ep), max(sp, ep))
            rng = high - low
            if rng <= 0 or trend not in ("UP", "DOWN"):
                level = None
            else:
                level = (high - 0.382 * rng) if trend == "UP" else (low + 0.382 * rng)

        # am Arm ablegen (non-breaking)
        try:
            setattr(arm, "fib382", None if level is None else float(level))
        except Exception:
            pass

        out.append({"arm_idx": idx, "start_idx": s_idx, "end_idx": e_idx, "level": level})

    return out


def kerzenfarbe(open_, close_):
    if close_ > open_:
        return "bullish"
    elif close_ < open_:
        return "bearish"
    else:
        return "doji"


def canonicalize_to_ha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Setzt Open/High/Low/Close auf HA_* und bewahrt vorhandene Raw-OHLC als Raw_*.
    Markiert den Frame als HA-Mode.
    """
    need = {"HA_Open", "HA_High", "HA_Low", "HA_Close"}
    if not need.issubset(df.columns):
        raise ValueError("canonicalize_to_ha erwartet HA_Open/HA_High/HA_Low/HA_Close im DataFrame.")

    df = df.copy()
    # Raw sichern, falls vorhanden (nur einmal anlegen)
    for base in ("Open", "High", "Low", "Close"):
        raw_col = f"Raw_{base}"
        if base in df.columns and raw_col not in df.columns:
            df[raw_col] = df[base]

    # Kanonisieren: O/H/L/C = HA_*
    df["Open"]  = df["HA_Open"]
    df["High"]  = df["HA_High"]
    df["Low"]   = df["HA_Low"]
    df["Close"] = df["HA_Close"]

    # Markierung
    df.attrs["data_mode"] = "HA"
    return df


def require_ha_mode(df: pd.DataFrame) -> None:
    """
    Stellt sicher, dass Funktionen NUR mit HA arbeiten.
    - data_mode == 'HA'
    - Open/High/Low/Close identisch zu HA_*
    """
    if df is None or len(df) == 0:
        raise ValueError("require_ha_mode: Leer/None DataFrame.")
    if df.attrs.get("data_mode") != "HA":
        raise ValueError("require_ha_mode: DataFrame nicht im HA-Mode. canonicalize_to_ha() zuerst aufrufen.")

    need = {"HA_Open", "HA_High", "HA_Low", "HA_Close"}
    if not need.issubset(df.columns):
        raise ValueError("require_ha_mode: HA-Spalten fehlen.")

    # Gleichheit prüfen (ohne NaN-Ärger)
    if not (df["Open"].equals(df["HA_Open"]) and
            df["High"].equals(df["HA_High"]) and
            df["Low"].equals(df["HA_Low"]) and
            df["Close"].equals(df["HA_Close"])):
        raise ValueError("require_ha_mode: O/H/L/C sind nicht identisch zu HA_*. Pipeline verletzt das HA-only-Versprechen.")


def remove_isolated_candles(df: pd.DataFrame) -> pd.DataFrame:
    # Guards
    if df is None or len(df) < 3:
        return df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else df

    # Primär HA  Fallback auf klassische OHLC, mit einmaligem Hinweis
    if {"HA_Open", "HA_Close"}.issubset(df.columns):
        o_col, c_col = "HA_Open", "HA_Close"
    elif {"Open", "Close"}.issubset(df.columns):
        o_col, c_col = "Open", "Close"
        print("?? remove_isolated_candles: HA_Open/HA_Close nicht gefunden  Fallback auf Open/Close.")
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

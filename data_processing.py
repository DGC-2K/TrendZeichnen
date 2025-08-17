import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta 
import numpy as np
import pandas.api.types as pd_types


def download_data(ticker: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
    """
    Lädt Finanzdaten für den angegebenen Ticker und Zeitraum von Yahoo Finance und bereinigt sie.
    """
    print(f"⬇️ Lade Daten für {ticker} von {start_date.strftime('%Y-%m-%d %H:%M')} bis {end_date.strftime('%Y-%m-%d %H:%M')}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            print(f"⚠️ Keine Daten für {ticker} im angegebenen Zeitraum und Intervall gefunden.")
            return pd.DataFrame() 
            
        # MultiIndex-Spalten beheben, falls vorhanden
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Spalte 'Adj Close' zu 'Close' umbenennen
        if 'Adj Close' in data.columns:
            data.rename(columns={'Adj Close': 'Close'}, inplace=True)
            
        required_ohlc_cols = ['Open', 'High', 'Low', 'Close'] 
        
        # Sicherstellen, dass alle grundlegenden OHLV-Spalten vorhanden sind
        if not all(col in data.columns for col in required_ohlc_cols):
            print(f"❌ Fehler: Eine der erforderlichen Spalten {required_ohlc_cols} fehlt in den Daten.")
            return pd.DataFrame() 

        # Konvertiere alle OHLV-Spalten zu numerischen Typen und entferne NaN-Werte
        for col in required_ohlc_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data.dropna(subset=required_ohlc_cols, inplace=True)
        
        if data.empty:
            print(f"⚠️ Daten für {ticker} nach der Bereinigung leer geworden.")
            return pd.DataFrame()

        # Index und 'Zeit' Spalte verarbeiten
        data.reset_index(inplace=True)
        
        if 'index' in data.columns:
            data.rename(columns={'index': 'Zeit'}, inplace=True) 
        elif 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Zeit'}, inplace=True) 
        else:
            print("❌ Fehler: Unbekannter Zeitstempel-Spaltenname.")
            return pd.DataFrame() 

        # 'Kerze_Nr' erstellen
        data['Kerze_Nr'] = range(len(data))
        
        # Endgültige Spaltenreihenfolge
        final_cols_order = ['Kerze_Nr', 'Zeit', 'Open', 'High', 'Low', 'Close']
        data = data[final_cols_order]

        print(f"✅ Daten erfolgreich geladen und vorstrukturiert. {len(data)} Kerzen.")
        return data

    except Exception as e:
        print(f"❌ Fehler beim Laden/Vorbereiten der Daten von yfinance: {e}")
        return pd.DataFrame()


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Bereitet die geladenen Daten auf, indem der Index gesetzt wird und letzte Typenprüfungen erfolgen.
    """
    if data.empty:
        return pd.DataFrame() 
    
    df_copy = data.copy()
    
    # Sicherstellen, dass 'Kerze_Nr' als Index gesetzt ist
    if 'Kerze_Nr' in df_copy.columns:
        if not df_copy['Kerze_Nr'].is_unique:
            df_copy['Kerze_Nr'] = range(len(df_copy))
            
        df_copy.set_index('Kerze_Nr', inplace=True)
    else:
        print("⚠️ Fehler: 'Kerze_Nr' Spalte nicht gefunden.")
        df_copy.index = pd.RangeIndex(start=0, stop=len(df_copy), name='Kerze_Nr')
        
    required_ohlc_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_ohlc_cols:
        if col in df_copy.columns and not pd_types.is_numeric_dtype(df_copy[col]): 
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        elif col not in df_copy.columns:
            df_copy[col] = np.nan 
            
    df_copy.dropna(subset=required_ohlc_cols, inplace=True)
    
    if df_copy.empty:
        print("⚠️ Warnung: DataFrame ist nach abschließender Bereinigung leer geworden.")

    return df_copy


def bestimme_heikin_ashi_farbe(df: pd.DataFrame, koerper_schwelle: float = 1.0) -> pd.Series:
    """
    Setzt die Farbe einer Heikin-Ashi-Kerze nach folgender Regel:
    - Wenn abs(Open - Close) > koerper_schwelle: normal (gruen/rot)
    - Wenn abs(Open - Close) <= koerper_schwelle: übernimmt die Farbe der ersten nachfolgenden Kerze mit abs(Open - Close) > koerper_schwelle.
      Falls es keine solche Kerze gibt, wird rückwärts gesucht.
      Falls auch dann keine gefunden wird, wird 'gruen' gesetzt.
    Gibt eine pd.Series mit den Farben ('gruen', 'rot') zurück.
    """
    farben = [None] * len(df)
    n = len(df)
    open_arr = df['Open'].values
    close_arr = df['Close'].values

    for i in range(n):
        koerper = abs(open_arr[i] - close_arr[i])
        if koerper > koerper_schwelle:
            farben[i] = "gruen" if close_arr[i] > open_arr[i] else "rot"
        else:
            # Suche erste nachfolgende Kerze mit Körper > Schwelle
            farbe_gesetzt = False
            for j in range(i + 1, n):
                koerper_j = abs(open_arr[j] - close_arr[j])
                if koerper_j > koerper_schwelle:
                    farben[i] = "gruen" if close_arr[j] > open_arr[j] else "rot"
                    farbe_gesetzt = True
                    break
            if not farbe_gesetzt:
                # Falls keine spätere Kerze mit Körper > Schwelle gefunden, rückwärts suchen
                for j in range(i - 1, -1, -1):
                    koerper_j = abs(open_arr[j] - close_arr[j])
                    if koerper_j > koerper_schwelle:
                        farben[i] = "gruen" if close_arr[j] > open_arr[j] else "rot"
                        farbe_gesetzt = True
                        break
            if not farbe_gesetzt:
                # Default (z.B. für komplett flachen Chart)
                farben[i] = "gruen"
    return pd.Series(farben, index=df.index)
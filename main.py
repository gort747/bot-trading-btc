import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

def enviar_telegram(mensaje):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def ejecutar_estrategia():
    print("Descargando datos recientes (60 días)...")
    df_live = yf.download("BTC-USD", period="60d", progress=False)
    
    if isinstance(df_live.columns, pd.MultiIndex):
        df_live.columns = df_live.columns.droplevel(1)
        
    df_live = df_live.dropna()

    print("Calculando características de micro-momentum del equipo...")
    df_live['Return'] = df_live['Close'].pct_change()
    
    df_live['EMA_3'] = df_live['Close'].ewm(span=3, adjust=False).mean()
    df_live['EMA_7'] = df_live['Close'].ewm(span=7, adjust=False).mean()
    df_live['EMA_Cross'] = np.where(df_live['EMA_3'] > df_live['EMA_7'], 1, -1)

    delta = df_live['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / (loss + 1e-9)
    df_live['RSI_7'] = 100 - (100 / (1 + rs))

    df_live['Vol_Rel_5d'] = df_live['Volume'] / df_live['Volume'].rolling(window=5).mean()

    high_low = df_live['High'] - df_live['Low']
    high_close = np.abs(df_live['High'] - df_live['Close'].shift())
    low_close = np.abs(df_live['Low'] - df_live['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_live['ATR_5'] = true_range.rolling(5).mean()
    df_live['ATR_Norm'] = df_live['ATR_5'] / df_live['Close'] 

    df_live['SMA_50'] = df_live['Close'].rolling(window=50).mean()

    df_live = df_live.dropna()
    hoy = df_live.iloc[-1:]

    precio_actual = float(hoy['Close'].values[0])
    sma_50_actual = float(hoy['SMA_50'].values[0])
    regimen_alcista = precio_actual > sma_50_actual

    print("Cargando el cerebro de la red neuronal (XGBoost)...")
    modelo_xgb = joblib.load('xgboost_btc_short_term_v1.pkl')

    features = ['Return', 'EMA_Cross', 'RSI_7', 'Vol_Rel_5d', 'ATR_Norm']
    X_live = hoy[features]
    
    prob_alcista = float(modelo_xgb.predict_proba(X_live)[:, 1][0])
    umbral = 0.56
    senal_xgboost = prob_alcista > umbral

    print("Construyendo y enviando reporte...")
    mensaje = f"🤖 *REPORTE DIARIO EQUIPO (XGBoost Corto Plazo)* 🤖\n\n"
    mensaje += f"💵 *Precio Actual BTC:* ${precio_actual:,.2f}\n"
    mensaje += f"📈 *Filtro Macro (SMA50):* {'ALCISTA 🟢' if regimen_alcista else 'BAJISTA 🔴'}\n"
    mensaje += f"🧠 *Certeza XGBoost:* {prob_alcista * 100:.2f}% (Req > {umbral*100}%)\n\n"
    
    if regimen_alcista and senal_xgboost:
        mensaje += "🚀 *SEÑAL:* 🟢 COMPRAR. Condiciones técnicas y macro alineadas."
    elif regimen_alcista and not senal_xgboost:
        mensaje += "🟡 *SEÑAL:* ESPERAR. Mercado alcista, pero sin momentum corto plazo."
    else:
        mensaje += "🛑 *SEÑAL:* QUEDARSE EN USDT. Riesgo estructural alto."
        
    enviar_telegram(mensaje)
    print("Proceso finalizado con éxito.")

if __name__ == "__main__":
    ejecutar_estrategia()

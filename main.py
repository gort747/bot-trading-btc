import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb

def enviar_telegram(mensaje):
    # Lee las credenciales de forma segura desde la nube
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def ejecutar_estrategia():
    print("Descargando datos históricos...")
    btc = yf.download('BTC-USD', start='2020-01-01', progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        btc[col] = btc[col].astype(float)
        
    print("Calculando características...")
    btc['ret_1d'] = btc['Close'].pct_change(1)
    btc['ret_3d'] = btc['Close'].pct_change(3)
    btc['ret_5d'] = btc['Close'].pct_change(5)
    btc['vol_14d'] = btc['ret_1d'].rolling(14).std()
    btc['sma_50'] = btc['Close'].rolling(window=50).mean()
    btc['sma_200'] = btc['Close'].rolling(window=200).mean()
    btc['dist_sma_50'] = (btc['Close'] - btc['sma_50']) / btc['sma_50']
    
    delta = btc['Close'].diff()
    ganancia = delta.where(delta > 0, 0).rolling(window=14).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganancia / (perdida + 1e-9)
    btc['rsi_14'] = 100 - (100 / (1 + rs))
    btc['macro_trend'] = (btc['Close'] > btc['sma_200']).astype(int)
    
    HORIZONTE = 3
    btc['Target'] = (btc['Close'].shift(-HORIZONTE) > btc['Close']).astype(int)
    
    # Separar datos de entrenamiento (quitamos los últimos 3 días sin target)
    btc_train = btc.dropna().copy()
    features = ['ret_1d', 'ret_3d', 'ret_5d', 'vol_14d', 'dist_sma_50', 'rsi_14', 'macro_trend']
    X_train = btc_train[features]
    y_train = btc_train['Target']
    
    print("Entrenando XGBoost...")
    modelo_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03, 
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    modelo_xgb.fit(X_train, y_train)
    
    print("Generando predicción de hoy...")
    ultima_fila = btc[features].iloc[-1:]
    precio_actual = float(btc['Close'].iloc[-1])
    es_macro_alcista = bool(btc['macro_trend'].iloc[-1] == 1)
    prob_alcista = float(modelo_xgb.predict_proba(ultima_fila)[:, 1][0])
    
    mensaje = f"🤖 *REPORTE DIARIO EQUIPO (XGBoost 3D)* 🤖\n\n"
    mensaje += f"💵 *Precio Actual BTC:* ${precio_actual:,.2f}\n"
    mensaje += f"📈 *Tendencia Macro (SMA200):* {'ALCISTA 🟢' if es_macro_alcista else 'BAJISTA 🔴'}\n"
    mensaje += f"🧠 *Confianza del Modelo:* {prob_alcista * 100:.1f}%\n\n"
    
    if prob_alcista >= 0.55 and es_macro_alcista:
        mensaje += "🚀 *SEÑAL:* EJECUTAR COMPRA (Entrar a 3 días)"
    elif prob_alcista < 0.50:
        mensaje += "🛑 *SEÑAL:* VENDER / PERMANECER EN DÓLARES"
    else:
        mensaje += "⏳ *SEÑAL:* NEUTRAL (Esperar confirmación)"
        
    enviar_telegram(mensaje)
    print("Proceso finalizado. Mensaje enviado.")

if __name__ == "__main__":
    ejecutar_estrategia()

import joblib
import numpy as np
import time
import psutil
import speedtest
from ping3 import ping
from datetime import datetime
import pandas as pd
import os

# Safely load available models
models = []

# RandomForest model
rf_model = joblib.load("network_failure_randomforest_model.pkl")
models.append(rf_model)

# ExtraTrees model
if os.path.exists("network_failure_extratrees_model.pkl"):
    et_model = joblib.load("network_failure_extratrees_model.pkl")
    models.append(et_model)
else:
    print("⚠️ ExtraTrees model not found. Skipping.")

# LightGBM model
if os.path.exists("network_failure_lightgbm_model.pkl"):
    lgbm_model = joblib.load("network_failure_lightgbm_model.pkl")
    models.append(lgbm_model)
else:
    print("⚠️ LightGBM model not found. Skipping.")

def get_network_status():
    """Check if network is connected and return current network stats."""
    try:
        net_io = psutil.net_io_counters()
        if net_io.bytes_sent == 0 and net_io.bytes_recv == 0:
            print(f"[{datetime.now()}] 🚨 Network Disconnected! No data transfer.")
            return None  

        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1e6  # Mbps
        upload_speed = st.upload() / 1e6  # Mbps
        
        latency = ping('8.8.8.8')
        latency = 9999 if latency is None else latency * 1000  # ms

        return [download_speed, upload_speed, latency, net_io.bytes_sent / 1e6, net_io.bytes_recv / 1e6]

    except Exception as e:
        print(f"[{datetime.now()}] Error collecting network data: {e}")
        return None

def predict_failure():
    """Predict network failure using ensemble voting."""
    stats = get_network_status()
    
    if stats is None:
        print(f"[{datetime.now()}] 🚨 Network failure detected (No connection).")
        return

    feature_columns = ["download_speed", "upload_speed", "latency", "data_sent", "data_received"]
    live_data = pd.DataFrame([stats], columns=feature_columns)

    votes = []

    for model in models:
        pred = model.predict(live_data)[0]
        votes.append(int(pred))  # cleaner output

    vote_sum = sum(votes)

    # Ensemble decision: if majority models predict failure
    if vote_sum >= (len(models) // 2) + 1:
        print(f"[{datetime.now()}] 🚨 ALERT: Network failure expected soon! (Votes: {votes})")
    else:
        print(f"[{datetime.now()}] ✅ Network is stable. (Votes: {votes})")

try:
    while True:
        predict_failure()
        time.sleep(5)
except KeyboardInterrupt:
    print("\n🛑 Monitoring stopped by user.")

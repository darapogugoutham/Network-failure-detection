import psutil
import speedtest
from ping3 import ping
import csv
import time
import random  # Used to simulate failures (Replace with real failure detection logic)

# Define CSV file to store data
csv_filename = "network_data.csv"

# Initialize CSV file with headers
with open(csv_filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["download_speed", "upload_speed", "latency", "data_sent", "data_received", "failure"])

def get_network_stats():
    try:
        # Measure network speed
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1e6  # Convert to Mbps
        upload_speed = st.upload() / 1e6  # Convert to Mbps
        
        # Measure network latency (ping Google)
        latency = ping('8.8.8.8') * 1000  # Convert to milliseconds
        
        # Get real-time data usage
        net_io = psutil.net_io_counters()
        sent = net_io.bytes_sent / 1e6  # Convert to MB
        received = net_io.bytes_recv / 1e6  # Convert to MB

        # Simulate network failure label (1 = failure, 0 = no failure)
        # Replace this with actual failure detection based on your criteria
        failure = 1 if latency > 500 or download_speed < 2 else 0

        return [download_speed, upload_speed, latency, sent, received, failure]
    
    except Exception as e:
        print(f"Error collecting network data: {e}")
        return None

# Collect data every 10 seconds
while True:
    data = get_network_stats()
    if data:
        with open(csv_filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
        print(f"Logged: {data}")

    time.sleep(5)  # Collect data every 10 seconds

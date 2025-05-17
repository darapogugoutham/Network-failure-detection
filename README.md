# Network-failure-detection

Network Failure Detection
=========================

Project Overview:
-----------------
This project focuses on detecting failures in a network using machine learning techniques. It collects real-time or historical network data (e.g., download/upload speed, latency, data usage) and uses trained classification models to predict potential failures.

The pipeline includes:
- Data collection and processing
- Model training and validation
- Failure prediction and evaluation

Project Structure:
------------------

1. collect_network_data.py
   - Simulates or gathers network data.
   - Outputs a CSV file: network_data.csv

2. train_model.py
   - Loads data from network_data.csv
   - Performs preprocessing and splits the data (train/val/test)
   - Trains classification models (e.g., Random Forest, LightGBM)
   - Saves the best performing model as model.pkl

3. predict_network.py
   - Loads model.pkl
   - Takes new network stats as input and predicts failure status
   - Displays prediction result

4. read_data.py
   - Loads and displays basic statistics about network_data.csv
   - Checks for nulls, schema, and label distribution

Dependencies:
-------------
- Python 3.x
- pandas
- numpy
- scikit-learn
- lightgbm
- joblib

Installation:
-------------
Use pip to install dependencies:

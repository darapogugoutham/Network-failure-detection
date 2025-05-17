import pandas as pd
import numpy as np
import joblib
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs("eval_visuals", exist_ok=True)

# Load dataset
data = pd.read_csv("network_data.csv")
X = data[["download_speed", "upload_speed", "latency", "data_sent", "data_received"]]
y = data["failure"]

# 70:20:10 split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=10, learning_rate=0.1,
        class_weight='balanced', random_state=42,
        verbosity=-1
    )
}

metrics_data = []
ensemble_votes = []

# Train each model and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, zero_division=1)

    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, zero_division=1)

    print(f"\n✅ {name} Validation F1 Score: {val_f1:.4f}")
    print(f"📊 {name} Test Accuracy: {test_accuracy:.4f}")
    print(f"📈 {name} Test F1 Score: {test_f1:.4f}")

    metrics_data.append([name, f"{test_accuracy:.4f}", f"{test_f1:.4f}"])
    ensemble_votes.append(test_pred)

    # Save the model
    joblib.dump(model, f"network_failure_{name.lower()}_model.pkl")
    print(f"💾 Saved {name} model as 'network_failure_{name.lower()}_model.pkl'")

# ===== Ensemble Evaluation =====
ensemble_votes = np.array(ensemble_votes)
ensemble_pred = np.round(np.mean(ensemble_votes, axis=0)).astype(int)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=1)
print(f"\n🧠 Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
print(f"⭐ Ensemble Model F1 Score: {ensemble_f1:.4f}")
metrics_data.append(["Ensemble", f"{ensemble_accuracy:.4f}", f"{ensemble_f1:.4f}"])

# ===== Save Metrics Table as JPG =====
metrics_df = pd.DataFrame(metrics_data, columns=["Model", "Accuracy", "F1 Score"])
plt.figure(figsize=(6, 2 + 0.4 * len(metrics_df)))
sns.set(font_scale=1.1)
sns.heatmap(metrics_df.set_index("Model").astype(float), annot=True, fmt=".4f", cmap="YlGnBu", cbar=False)
plt.title("Model Evaluation Metrics")
plt.tight_layout()
plt.savefig("eval_visuals/ensemble_model_metrics.jpg")
plt.close()

print("\n🖼️ Metrics table image saved to 'eval_visuals/ensemble_model_metrics.jpg'")
print("✅✅✅ All models trained, evaluated, and saved successfully!")

# ============================================================
#  Project 2: Data Classification Using AI
#  Algorithm : K-Nearest Neighbors (KNN)
#  Dataset   : Iris Benchmark
#  Author    : Mumiz Ahmad | mumizahmad30@gmail.com
# ============================================================

# ---------- 1. IMPORTS ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)

# ============================================================
# STEP 1 — LOAD & UNDERSTAND THE DATASET
# ============================================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print("  PROJECT 2 — DATA CLASSIFICATION USING AI (KNN)")
print("=" * 55)

print("\n📦 Dataset Shape :", df.shape)
print("\n🔍 First 5 Rows:")
print(df.head())
print("\n📊 Class Distribution:")
print(df["species"].value_counts())
print("\n📈 Statistical Summary:")
print(df.describe().round(2))

# ============================================================
# STEP 2 — FEATURE SCALING  (Gatekeeper Rule)
# ============================================================
X = iris.data          # Features  (sepal_len, sepal_wid, petal_len, petal_wid)
y = iris.target        # Labels    (0=setosa, 1=versicolor, 2=virginica)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Feature Scaling Applied  →  Mean=0, Variance=1")

# ============================================================
# STEP 3 — TRAIN / TEST SPLIT  (80 / 20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n📂 Training Samples : {len(X_train)}")
print(f"📂 Testing  Samples : {len(X_test)}")

# ============================================================
# STEP 4 — FIND OPTIMAL K  (Elbow Method)
# ============================================================
error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

optimal_k = k_range[np.argmin(error_rates)]
print(f"\n🎯 Optimal K (Elbow) : {optimal_k}")

# ============================================================
# STEP 5 — TRAIN FINAL MODEL
# ============================================================
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)                       # FIT  — Memorize the map
predictions = model.predict(X_test)               # PREDICT — Apply logic

# ============================================================
# STEP 6 — OUTPUT VALIDATION
# ============================================================
acc = accuracy_score(y_test, predictions)
f1  = f1_score(y_test, predictions, average="weighted")

print("\n" + "=" * 55)
print("  OUTPUT VALIDATION")
print("=" * 55)
print(f"\n🏆 Accuracy  : {acc * 100:.2f}%")
print(f"📐 F1 Score  : {f1:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# ============================================================
# STEP 7 — VISUALIZATIONS  (4 plots saved as one figure)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Project 2 — KNN Iris Classification Dashboard", fontsize=15, fontweight="bold")

# --- Plot A: Confusion Matrix ---
cm = confusion_matrix(y_test, predictions)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=axes[0, 0],
)
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_xlabel("Predicted Label")
axes[0, 0].set_ylabel("True Label")

# --- Plot B: Elbow Curve (Choosing K) ---
axes[0, 1].plot(k_range, error_rates, marker="o", color="steelblue", linewidth=2)
axes[0, 1].axvline(x=optimal_k, color="orange", linestyle="--", label=f"Optimal K={optimal_k}")
axes[0, 1].set_title("Elbow Method — Choosing Optimal K")
axes[0, 1].set_xlabel("K Value")
axes[0, 1].set_ylabel("Error Rate")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# --- Plot C: Feature Distribution (Petal Length vs Petal Width) ---
colors = ["#E74C3C", "#3498DB", "#2ECC71"]
for i, (name, color) in enumerate(zip(iris.target_names, colors)):
    mask = y == i
    axes[1, 0].scatter(
        X[mask, 2], X[mask, 3],
        label=name, color=color, alpha=0.7, edgecolors="k", linewidths=0.4
    )
axes[1, 0].set_title("Feature Space: Petal Length vs Petal Width")
axes[1, 0].set_xlabel("Petal Length (cm)")
axes[1, 0].set_ylabel("Petal Width (cm)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# --- Plot D: Per-Class F1 Score Bar Chart ---
report = classification_report(
    y_test, predictions, target_names=iris.target_names, output_dict=True
)
classes   = iris.target_names
f1_scores = [report[c]["f1-score"] for c in classes]
bar_colors = ["#E74C3C", "#3498DB", "#2ECC71"]
bars = axes[1, 1].bar(classes, f1_scores, color=bar_colors, edgecolor="black", alpha=0.85)
axes[1, 1].set_title("Per-Class F1 Score")
axes[1, 1].set_ylabel("F1 Score")
axes[1, 1].set_ylim(0, 1.1)
for bar, score in zip(bars, f1_scores):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{score:.2f}", ha="center", fontweight="bold"
    )
axes[1, 1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("knn_dashboard.png", dpi=150, bbox_inches="tight")
print("\n📊 Dashboard saved  →  knn_dashboard.png")

# ============================================================
# STEP 8 — LIVE PREDICTION DEMO
# ============================================================
print("\n" + "=" * 55)
print("  LIVE PREDICTION DEMO")
print("=" * 55)

sample = np.array([[5.1, 3.5, 1.4, 0.2]])          # Classic Setosa sample
sample_scaled = scaler.transform(sample)
pred_label = model.predict(sample_scaled)[0]

print(f"\nInput  →  Sepal: 5.1 x 3.5 cm  |  Petal: 1.4 x 0.2 cm")
print(f"Output →  Predicted Species : {iris.target_names[pred_label].upper()}")
print("\n✅ Project 2 Complete!\n")

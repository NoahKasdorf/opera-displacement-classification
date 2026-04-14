import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


data_path = Path("../processed/features_all.csv")
df = pd.read_csv(data_path)


hold_out_regions = ["san_joaquin", "hayward", "washington"]

# Manual Split by Region (Replacing train_test_split)
train_df = df[~df["region"].isin(hold_out_regions)]
test_df = df[df["region"].isin(hold_out_regions)]

print(f"Training on: {train_df['region'].unique()}")
print(f"Testing on:  {test_df['region'].unique()}")

# Separate Features (X) and Labels (y)
meta_cols = ["pixel_y", "pixel_x", "frame_id", "region", "label"]

X_train = train_df.drop(columns=meta_cols)
y_train = train_df["label"]

X_test = test_df.drop(columns=meta_cols)
y_test = test_df["label"]

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Multinomial Model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
model.fit(X_train_scaled, y_train)

# Performance Evaluation
y_pred = model.predict(X_test_scaled)
classes = model.classes_

print("\n" + "=" * 40)
print("GENERALIZATION REPORT (HOLD-OUT REGIONS)")
print("=" * 40)
print(classification_report(y_test, y_pred))

# Multi-Class Feature Importance
importance_df = pd.DataFrame(model.coef_, columns=X_train.columns, index=classes)

print("\n" + "=" * 40)
print("UNIVERSAL FEATURES (Learned from Training Regions)")
print("=" * 40)
for cls in classes:
    top_feat = importance_df.loc[cls].idxmax()
    print(f"[{cls.upper()}] Strongest Predictor: {top_feat}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Purples"
)
plt.title("Generalization Performance on Unseen Regions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



from sklearn.model_selection import train_test_split



data_path = Path("../processed/features_all.csv")
df = pd.read_csv(data_path)


meta_cols = ["pixel_y", "pixel_x", "frame_id", "region", "label"]
X = df.drop(columns=meta_cols)
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Total samples: {len(df)}")
print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
classes = model.classes_

print("\n" + "=" * 40)
print("SHUFFLED 80/20 CLASSIFICATION REPORT")
print("=" * 40)
print(classification_report(y_test, y_pred))

importance_df = pd.DataFrame(model.coef_, columns=X.columns, index=classes)

print("\n" + "=" * 40)
print("TOP FEATURES BY CLASS")
print("=" * 40)
for cls in classes:
    top_feat = importance_df.loc[cls].idxmax()
    print(
        f"[{cls.upper()}] Strongest Predictor: {top_feat} ({importance_df.loc[cls, top_feat]:.3f})"
    )

# Visualizing the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Purples"
)
plt.title("80/20 Shuffle Split Performance")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

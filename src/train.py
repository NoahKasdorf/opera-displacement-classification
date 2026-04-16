"""
OPERA DISP-S1 Classification: Training Pipeline

Trains LR, a hybrid 1D CNN, and (optionally) XGBoost and KNN+PCA on the
displacement feature set and raw time series. The 80/20 stratified split is
saved so teammates can reproduce the exact same evaluation.

Usage:
    cd src && python train.py

Outputs:
    results/predictions.npz  (labels, predictions, embeddings, metrics)
    models/lr_model.joblib
    models/cnn_model.weights.h5
    models/xgb_*.joblib  (only if RUN_XGB is True)
"""

# Keras backend must be set before importing keras.
# TF is broken with NumPy 2.x, so we force the PyTorch backend.
import os

os.environ["KERAS_BACKEND"] = "torch"

# Windows terminals default to CP-1252 which chokes on Unicode print statements.
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import keras
from keras import layers
from config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR,
    META_COLS, TEMPORAL_KEYWORDS, SPATIAL_PREFIXES,
    RANDOM_SEED, TEST_SIZE, HOLDOUT_REGIONS,
    CNN_SEQ_LEN, CNN_EPOCHS, CNN_BATCH_SIZE, CNN_VAL_SPLIT, EARLY_STOP_PATIENCE,
)
from sklearn.pipeline import Pipeline

RUN_XGB = True
RUN_KNN = True


np.random.seed(RANDOM_SEED)
try:
    # keras.utils.set_random_seed covers all backends, but also touches TF.
    # TF is broken here (NumPy 2.x conflict), so we fall back to PyTorch seeding.
    keras.utils.set_random_seed(RANDOM_SEED)
except Exception:
    import torch
    torch.manual_seed(RANDOM_SEED)


# Feature split

def classify_features(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Splits columns into temporal and spatial/DEM groups.

    Temporal: name contains a TEMPORAL_KEYWORD and doesn't start with SPATIAL_PREFIX.
    Everything else goes to spatial/DEM.

    Returns (temporal_cols, spatial_cols).
    """
    temporal, spatial = [], []
    for col in feature_cols:
        if col.startswith(SPATIAL_PREFIXES):
            spatial.append(col)
        elif any(kw in col for kw in TEMPORAL_KEYWORDS):
            temporal.append(col)
        else:
            spatial.append(col)
    return temporal, spatial


def build_regional_split(
    df: pd.DataFrame, y_enc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Geographic train/test split holding out one region per class.

    Test set: HOLDOUT_REGIONS (one per class). Train set: everything else.

    Returns (reg_train_idx, reg_test_idx) as indices into the full dataset.
    """
    regions = df["region"].values
    labels  = df["label"].values

    test_mask = np.zeros(len(df), dtype=bool)
    for cls, region in HOLDOUT_REGIONS.items():
        test_mask |= (regions == region) & (labels == cls)

    return np.where(~test_mask)[0], np.where(test_mask)[0]


# Data loading

def load_features(csv_path: Path) -> tuple:
    """Loads features_all.csv and imputes NaN values with column medians.

    Returns (df, X as float32, y_str label strings, feat_cols list).
    """
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feat_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return df, X.values.astype(np.float32), df["label"].values, feat_cols


def load_all_timeseries(df: pd.DataFrame, seq_len: int = CNN_SEQ_LEN) -> np.ndarray:
    """Assembles per-region NPZ files into a single matrix aligned with df.

    Series are truncated to seq_len timesteps or zero-padded if shorter.
    Rows are matched by (region, pixel_y, pixel_x).

    Returns an array of shape (N, seq_len), float32.
    """
    print("  Building (region, pixel_y, pixel_x) → time_series lookup …")
    ts_lookup: dict[tuple, np.ndarray] = {}
    for region in df["region"].unique():
        npz_path = PROCESSED_DIR / f"timeseries_{region}.npz"
        if not npz_path.exists():
            print(f"  WARNING: {npz_path} not found, filling with zeros")
            continue
        data = np.load(npz_path)
        ts = data["time_series"]              # (N_region, T)
        py = data["pixel_y"].astype(int)
        px = data["pixel_x"].astype(int)
        for i in range(len(py)):
            ts_lookup[(region, py[i], px[i])] = ts[i]

    N = len(df)
    out = np.zeros((N, seq_len), dtype=np.float32)
    missing = 0
    for i, row in enumerate(df.itertuples(index=False)):
        key = (row.region, int(row.pixel_y), int(row.pixel_x))
        if key in ts_lookup:
            ts = ts_lookup[key]
            T = len(ts)
            copy_len = min(T, seq_len)
            out[i, :copy_len] = ts[:copy_len].astype(np.float32)
        else:
            missing += 1

    if missing:
        print(f"  WARNING: {missing}/{N} samples had no matching time series")
    return out


# CNN

def build_hybrid_cnn(seq_len: int, n_spatial: int, n_classes: int) -> keras.Model:
    """Hybrid 1D CNN fusing raw time series with spatial/DEM features.

    Temporal branch: three Conv1D layers (64, 128, 64) with max-pooling, then
    GlobalAvgPool and Dense(128) "embedding" layer (temporal-only, used for UMAP).

    Spatial branch: pre-scaled spatial/DEM features concatenated with the embedding,
    then Dense(64) and Dense(n_classes, softmax).

    Keeping spatial features out of the embedding lets UMAP panels stay comparable.
    """
    ts_input = keras.Input(shape=(seq_len, 1), name="ts_input")

    x = layers.Conv1D(64, kernel_size=7, activation="relu",
                      padding="same", name="conv1")(ts_input)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(128, kernel_size=5, activation="relu",
                      padding="same", name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = layers.Conv1D(64, kernel_size=3, activation="relu",
                      padding="same", name="conv3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    emb = layers.Dense(128, activation="relu", name="embedding")(x)

    sp_input = keras.Input(shape=(n_spatial,), name="spatial_input")

    merged  = layers.Concatenate(name="merged")([emb, sp_input])
    fused   = layers.Dense(64, activation="relu", name="fc1")(merged)
    outputs = layers.Dense(n_classes, activation="softmax", name="classifier")(fused)

    return keras.Model(inputs=[ts_input, sp_input], outputs=outputs,
                       name="hybrid_cnn")


def extract_embeddings(model: keras.Model, X_ts_norm: np.ndarray,
                       batch_size: int = 256) -> np.ndarray:
    """Extracts the 128-d temporal embedding from the CNN branch.

    Builds a sub-model from ts_input to the embedding layer, keeping spatial
    features out so the UMAP comparison stays honest.
    """
    # For the hybrid model, input is [ts_input, spatial_input]; use index [0].
    ts_tensor = model.input[0] if isinstance(model.input, list) else model.input
    embed_model = keras.Model(
        inputs=ts_tensor,
        outputs=model.get_layer("embedding").output,
    )
    return embed_model.predict(X_ts_norm, batch_size=batch_size, verbose=0)


# Training functions

def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label_classes: np.ndarray,
) -> tuple:
    """Trains Logistic Regression on the full engineered feature set.

    Applies StandardScaler and uses class_weight='balanced' for class imbalance.
    Returns (lr_model, scaler, test_preds).
    """
    print("\n── Logistic Regression ─────────────────────────────────────────────")
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
    )
    lr.fit(X_tr_sc, y_train)

    preds = lr.predict(X_te_sc)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"  Test accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=label_classes))

    return lr, scaler, preds


def train_cnn(
    X_ts_train: np.ndarray, y_train_cat: np.ndarray,
    X_ts_test: np.ndarray,  y_test_enc: np.ndarray,
    X_sp_train: np.ndarray, X_sp_test: np.ndarray,
    n_classes: int, seq_len: int, label_classes: np.ndarray,
) -> tuple:
    """Trains the hybrid 1D CNN with early stopping.

    Takes normalized time series and pre-scaled spatial features as two separate
    inputs. Time series are globally normalized on train only to avoid leakage.
    CNN embeddings come from the temporal branch only to keep UMAP fair.

    Returns (model, ts_norm_params, test_preds, emb_train, emb_test).
    """
    print("\n── Hybrid 1D CNN (temporal + spatial) ──────────────────────────────")

    # Normalize on train only to avoid leakage.
    ts_mean = float(X_ts_train.mean())
    ts_std  = float(X_ts_train.std()) + 1e-8
    X_tr_norm = ((X_ts_train - ts_mean) / ts_std)[:, :, np.newaxis]  # (N, T, 1)
    X_te_norm = ((X_ts_test  - ts_mean) / ts_std)[:, :, np.newaxis]

    n_spatial = X_sp_train.shape[1]
    model = build_hybrid_cnn(seq_len, n_spatial, n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"\n  Training on {len(X_tr_norm)} samples "
          f"(seq_len={seq_len}, n_spatial={n_spatial}, batch_size={CNN_BATCH_SIZE}) …")
    model.fit(
        [X_tr_norm, X_sp_train],
        y_train_cat,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_split=CNN_VAL_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    test_proba = model.predict([X_te_norm, X_sp_test], verbose=0)
    test_preds = np.argmax(test_proba, axis=1)
    acc = accuracy_score(y_test_enc, test_preds)
    f1  = f1_score(y_test_enc, test_preds, average="macro")
    print(f"\n  Test accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}")
    print(classification_report(y_test_enc, test_preds, target_names=label_classes))

    print("  Extracting CNN embeddings (temporal branch only) …")
    emb_test  = extract_embeddings(model, X_te_norm)
    emb_train = extract_embeddings(model, X_tr_norm)
    print(f"  Embedding shape: {emb_test.shape}")

    ts_norm_params = {"mean": ts_mean, "std": ts_std}
    return model, ts_norm_params, test_preds, emb_train, emb_test


def train_xgb_ablation(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    temporal_cols: list[str], spatial_cols: list[str],
    all_feat_cols: list[str], label_classes: np.ndarray) -> dict:
    """Trains XGBoost three times: temporal-only, spatial-only, and all features.

    Use compute_sample_weight('balanced') for class imbalance.
    Feature importance is computed only on the 'all' run.

    Returns a dict with keys:
        "temporal", "spatial", "all": each a dict with model, preds, acc, f1.
        "xgb_feature_names": array of str.
        "xgb_feature_importance": array of float (model.feature_importances_).
    """
    results = {}
    col_idx_map = {c: i for i, c in enumerate(all_feat_cols)}
    t_idx = [col_idx_map[c] for c in temporal_cols]
    s_idx = [col_idx_map[c] for c in spatial_cols]

    sw = compute_sample_weight("balanced", y_train)

    for run_name, col_indices in [("temporal", t_idx), ("spatial", s_idx),
                                      ("all", list(range(len(all_feat_cols))))]:
          X_tr = X_train[:, col_indices]
          X_te = X_test[:,  col_indices]
          clf  = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.01, objective="multi:softprob",
                                   subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                                   random_state=RANDOM_SEED, tree_method="hist")
          clf.fit(X_tr, y_train, sample_weight=sw)
          preds = clf.predict(X_te)
          results[run_name] = {
              "model": clf,
              "preds": preds,
              "acc": accuracy_score(y_test, preds),
              "f1": f1_score(y_test, preds, average="macro"),
          }

          if run_name == "all":
              results["xgb_feature_names"] = np.array(all_feat_cols)
              results["xgb_feature_importance"] = clf.feature_importances_

          print(run_name)
          print(classification_report(y_test, preds, target_names=label_classes))


    return results


def train_knn_pca(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray, label_classes: np.ndarray) -> dict:
    """KNN classifier on PCA-reduced features.

    Expects the full scaled feature matrix 

    Returns a dict with keys: preds, acc, f1, pca_embeddings_test,
    pca_embeddings_train, model, pca, n_components.
    """

    # define model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)),
        ("knn", KNeighborsClassifier(metric="euclidean", weights="distance")),
    ])

    # Cross-validate to find optimal n_neighbours
    params = {'knn__n_neighbors': [5, 7, 9, 11, 15, 21]}
    cv = GridSearchCV(model, params, scoring='f1_macro', cv=10)
    cv.fit(X_train, y_train)
    best_model = cv.best_estimator_ # cv.fit trained the model with the best n_neighbours

    # variables
    pca_components = best_model.named_steps['pca'].n_components_
    preds = best_model.predict(X_test)
    pre_knn = best_model[:-1] # grabs the model prior to running knn. used for pca_embeddings in results

    print(f"  PCA kept {pca_components} components")
    print(f" KNN using {cv.best_params_['knn__n_neighbors']} neighbours")

    results = {
        "preds": preds,
        "acc": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="macro"),
        "pca_embeddings_test": pre_knn.transform(X_test),
        "pca_embeddings_train": pre_knn.transform(X_train),
        "model": best_model.named_steps["knn"], # i think this is right? can just return best_model otherwise
        "pca": best_model.named_steps["pca"],
        "n_components": pca_components
    }

    print(classification_report(y_test, preds, target_names=label_classes))

    return results


def train_regional_evaluation(
    X_feat: np.ndarray, X_ts: np.ndarray, X_sp: np.ndarray,
    y_enc: np.ndarray, df: pd.DataFrame,
    feat_cols: list[str], label_classes: np.ndarray, n_classes: int,
) -> dict:
    """Retrains all models on the regional hold-out split.

    Every model is trained from scratch on regions outside HOLDOUT_REGIONS and
    tested only on held-out regions. Separate from the main stratified split.

    Returns a dict with keys: train_indices, test_indices, y_train, y_test,
    lr_preds/acc/f1, cnn_preds/acc/f1, xgb_all_preds/acc/f1.
    """
    print("\n── Regional Hold-out Evaluation ─────────────────────────────────────")
    print("  Hold-out regions:")
    for cls, region in HOLDOUT_REGIONS.items():
        print(f"    {cls:<12} -> {region}")

    reg_train_idx, reg_test_idx = build_regional_split(df, y_enc)

    X_tr, X_te = X_feat[reg_train_idx], X_feat[reg_test_idx]
    y_tr, y_te = y_enc[reg_train_idx],  y_enc[reg_test_idx]
    X_ts_tr    = X_ts[reg_train_idx]
    X_ts_te    = X_ts[reg_test_idx]

    # Scale spatial features on train only to avoid leakage.
    sp_scaler_reg = StandardScaler()
    X_sp_tr = sp_scaler_reg.fit_transform(X_sp[reg_train_idx].astype(np.float64)).astype(np.float32)
    X_sp_te = sp_scaler_reg.transform(X_sp[reg_test_idx].astype(np.float64)).astype(np.float32)

    print(f"\n  Regional train: {len(reg_train_idx)}  |  Regional test: {len(reg_test_idx)}")
    y_str = df["label"].values
    for cls in label_classes:
        n_tr = np.sum(y_str[reg_train_idx] == cls)
        n_te = np.sum(y_str[reg_test_idx]  == cls)
        print(f"    {cls:<12}  train={n_tr}  test={n_te}")

    landslide_train = np.sum(y_str[reg_train_idx] == "landslide")
    if landslide_train <= 5:
        print(f"\n  WARNING: only {landslide_train} landslide training sample(s) — "
              f"expect near-zero recall on this class. "
              f"This reflects the limited geographic diversity of landslide data.")

    # LR
    print("\n  [Regional] Logistic Regression ...")
    _, _, reg_lr_preds = train_logistic_regression(X_tr, y_tr, X_te, y_te, label_classes)
    reg_lr_acc = accuracy_score(y_te, reg_lr_preds)
    reg_lr_f1  = f1_score(y_te, reg_lr_preds, average="macro")

    # CNN
    print("\n  [Regional] Hybrid 1D CNN ...")
    y_tr_cat = keras.utils.to_categorical(y_tr, n_classes)
    _, _, reg_cnn_preds, _, _ = train_cnn(
        X_ts_tr, y_tr_cat, X_ts_te, y_te,
        X_sp_tr, X_sp_te,
        n_classes, CNN_SEQ_LEN, label_classes,
    )
    reg_cnn_acc = accuracy_score(y_te, reg_cnn_preds)
    reg_cnn_f1  = f1_score(y_te, reg_cnn_preds, average="macro")

    # XGBoost (all features only; ablation was done in the random split)
    if RUN_XGB:
        print("\n  [Regional] XGBoost (all features) ...")
        sw = compute_sample_weight("balanced", y_tr)
        xgb_reg = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=RANDOM_SEED,
            tree_method="hist", verbosity=0,
        )
        xgb_reg.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        reg_xgb_preds = xgb_reg.predict(X_te)
        reg_xgb_acc   = accuracy_score(y_te, reg_xgb_preds)
        reg_xgb_f1    = f1_score(y_te, reg_xgb_preds, average="macro")
    else:
        reg_xgb_preds = reg_xgb_acc = reg_xgb_f1 = None

    print(f"\n  Regional hold-out summary (macro F1 / accuracy):")
    print(f"    LR      : F1={reg_lr_f1:.4f}  acc={reg_lr_acc:.4f}")
    print(f"    CNN     : F1={reg_cnn_f1:.4f}  acc={reg_cnn_acc:.4f}")
    if RUN_XGB:
        print(f"    XGB-all : F1={reg_xgb_f1:.4f}  acc={reg_xgb_acc:.4f}")

    return dict(
        train_indices = reg_train_idx,
        test_indices  = reg_test_idx,
        y_train = y_tr, y_test = y_te,
        lr_preds   = reg_lr_preds,
        lr_acc     = reg_lr_acc,  lr_f1   = reg_lr_f1,
        cnn_preds  = reg_cnn_preds,
        cnn_acc    = reg_cnn_acc, cnn_f1  = reg_cnn_f1,
        xgb_all_preds = reg_xgb_preds,
        xgb_all_acc   = reg_xgb_acc, xgb_all_f1 = reg_xgb_f1,
    )


# Main

def main():
    """Runs the full training pipeline: LR, CNN, optional XGBoost, then regional hold-out."""
    print("=" * 68)
    print("OPERA DISP-S1  —  Training Pipeline")
    print("=" * 68)

    # Create output directories
    for d in (RESULTS_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Load features
    print("\n[1/6]  Loading engineered features …")
    csv_path = PROCESSED_DIR / "features_all.csv"
    df, X_feat, y_str, feat_cols = load_features(csv_path)
    print(f"  Feature matrix : {X_feat.shape}  ({len(feat_cols)} features)")
    print(f"  Label counts   :")
    for lbl, cnt in zip(*np.unique(y_str, return_counts=True)):
        print(f"    {lbl:<12}: {cnt}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    label_classes = le.classes_
    n_classes = len(label_classes)
    print(f"  Encoding       : {dict(zip(label_classes, range(n_classes)))}")

    # Load time series
    print(f"\n[2/6]  Loading raw time series (seq_len={CNN_SEQ_LEN}) …")
    X_ts = load_all_timeseries(df, seq_len=CNN_SEQ_LEN)
    print(f"  Time series matrix: {X_ts.shape}")
    ts_lengths = []
    for r in df["region"].unique():
        import numpy as _np
        d = _np.load(PROCESSED_DIR / f"timeseries_{r}.npz")
        ts_lengths.append(f"{r}(T={d['time_series'].shape[1]})")
    print("  Original lengths: " + ", ".join(ts_lengths))

    # Feature split
    print("\n[3/6]  Identifying temporal vs spatial/DEM feature columns …")
    temporal_cols, spatial_cols = classify_features(feat_cols)
    print(f"\n  Temporal features ({len(temporal_cols)}):")
    for c in temporal_cols:
        print(f"    {c}")
    print(f"\n  Spatial / DEM features ({len(spatial_cols)}):")
    for c in spatial_cols:
        print(f"    {c}")

    # Stratified train/test split
    print("\n[4/6]  Stratified train/test split (80/20, seed=42) …")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                  random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X_feat, y_enc))

    X_train, X_test = X_feat[train_idx], X_feat[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    X_ts_train, X_ts_test = X_ts[train_idx], X_ts[test_idx]

    # Spatial features for the hybrid CNN, scaler fit on train only
    s_col_idx    = [feat_cols.index(c) for c in spatial_cols]
    sp_scaler    = StandardScaler()
    X_sp_train   = sp_scaler.fit_transform(
                       X_train[:, s_col_idx].astype(np.float64)).astype(np.float32)
    X_sp_test    = sp_scaler.transform(
                       X_test[:, s_col_idx].astype(np.float64)).astype(np.float32)
    
    # Full-dataset spatial for regional eval 
    X_sp_all     = X_feat[:, s_col_idx]

    print(f"  Train: {len(train_idx)}  |  Test: {len(test_idx)}")
    for cls in label_classes:
        n_tr = np.sum(y_str[train_idx] == cls)
        n_te = np.sum(y_str[test_idx] == cls)
        print(f"    {cls:<12}  train={n_tr}  test={n_te}")

    y_train_cat = keras.utils.to_categorical(y_train, n_classes)
    y_test_cat  = keras.utils.to_categorical(y_test,  n_classes)

    #  Logistic Regression
    print("\n[5a/6]  Training Logistic Regression …")
    lr_model, lr_scaler, lr_preds = train_logistic_regression(
        X_train, y_train, X_test, y_test, label_classes
    )

    # Hybrid CNN
    print("\n[5b/6]  Training Hybrid 1D CNN …")
    cnn_model, ts_norm_params, cnn_preds, cnn_emb_train, cnn_emb_test = train_cnn(
        X_ts_train, y_train_cat,
        X_ts_test, y_test,
        X_sp_train, X_sp_test,
        n_classes, CNN_SEQ_LEN, label_classes,
    )

    # XGBoost ablation
    if RUN_XGB:
        print("\n[6/7]  XGBoost ablation study …")
        ablation = train_xgb_ablation(
            X_train, y_train, X_test, y_test,
            temporal_cols, spatial_cols, feat_cols, label_classes,
        )
    else:
        print("\n[6/7]  XGBoost ablation — skipped (RUN_XGB=False)")
        ablation = None

    # KNN + PCA
    if RUN_KNN:
        print("\n[6b/7]  KNN + PCA …")
        # KNN uses the same scaled all-feature matrix as LR
        knn_result = train_knn_pca(
            X_train, y_train, X_test, y_test, label_classes,
        )
    else:
        print("\n[6b/7]  KNN + PCA — skipped (RUN_KNN=False)")
        knn_result = None

    # Regional hold-out evaluation
    print("\n[7/7]  Regional hold-out (geographic generalisation) …")
    reg_eval = train_regional_evaluation(
        X_feat, X_ts, X_sp_all, y_enc, df, feat_cols, label_classes, n_classes,
    )

    # Save models
    print("\nSaving models …")
    joblib.dump({"model": lr_model, "scaler": lr_scaler},
                MODELS_DIR / "lr_model.joblib")
    print(f"  Saved: {MODELS_DIR / 'lr_model.joblib'}")

    # model.save() imports TF internally (broken on NumPy 2.x), so we save weights only.
    cnn_model.save_weights(str(MODELS_DIR / "cnn_model.weights.h5"))
    print(f"  Saved: {MODELS_DIR / 'cnn_model.weights.h5'}")

    if ablation is not None:
        for run_name in ("temporal", "spatial", "all"):
            path = MODELS_DIR / f"xgb_{run_name}.joblib"
            joblib.dump(ablation[run_name]["model"], path)
            print(f"  Saved: {path}")

   
    print("\nSaving results …")

    # Feature slices for test set 
    t_col_idx = [feat_cols.index(c) for c in temporal_cols]
    temporal_feat_test = X_test[:, t_col_idx]
    all_feat_test      = X_test

    out_path = RESULTS_DIR / "predictions.npz"

    save_dict = {
        # Labels and split indices
        "y_train"       : y_train,
        "y_test"        : y_test,
        "train_indices" : train_idx,
        "test_indices"  : test_idx,
        "label_classes" : label_classes,
        
        # Predictions 
        "lr_preds"  : lr_preds,
        "cnn_preds" : cnn_preds,
        
        # CNN embeddings 
        "cnn_embeddings_test"  : cnn_emb_test,
        "cnn_embeddings_train" : cnn_emb_train,
        
        # Feature matrices for UMAP panels 
        "temporal_feat_test"  : temporal_feat_test,  
        "all_feat_test"       : all_feat_test,         
        "temporal_feat_names" : np.array(temporal_cols),
        "all_feat_names"      : np.array(feat_cols),
        
        # CNN normalization params
        "cnn_ts_mean" : np.float32(ts_norm_params["mean"]),
        "cnn_ts_std"  : np.float32(ts_norm_params["std"]),
        "cnn_sp_mean" : sp_scaler.mean_.astype(np.float32),
        "cnn_sp_std"  : sp_scaler.scale_.astype(np.float32),
        
        #  Regional hold-out
        "regional_train_indices" : reg_eval["train_indices"],
        "regional_test_indices"  : reg_eval["test_indices"],
        "regional_y_train"       : reg_eval["y_train"],
        "regional_y_test"        : reg_eval["y_test"],
        "regional_lr_preds"      : reg_eval["lr_preds"],
        "regional_cnn_preds"     : reg_eval["cnn_preds"],
        "regional_lr_acc"        : np.float32(reg_eval["lr_acc"]),
        "regional_lr_f1"         : np.float32(reg_eval["lr_f1"]),
        "regional_cnn_acc"       : np.float32(reg_eval["cnn_acc"]),
        "regional_cnn_f1"        : np.float32(reg_eval["cnn_f1"]),
    }

    # only present when RUN_XGB=True 
    if ablation is not None:
        save_dict.update({
            "xgb_temporal_preds"     : ablation["temporal"]["preds"],
            "xgb_spatial_preds"      : ablation["spatial"]["preds"],
            "xgb_all_preds"          : ablation["all"]["preds"],
            "xgb_feature_names"      : ablation["xgb_feature_names"],
            "xgb_feature_importance" : ablation["xgb_feature_importance"],
            "ablation_temporal_acc"  : np.float32(ablation["temporal"]["acc"]),
            "ablation_temporal_f1"   : np.float32(ablation["temporal"]["f1"]),
            "ablation_spatial_acc"   : np.float32(ablation["spatial"]["acc"]),
            "ablation_spatial_f1"    : np.float32(ablation["spatial"]["f1"]),
            "ablation_all_acc"       : np.float32(ablation["all"]["acc"]),
            "ablation_all_f1"        : np.float32(ablation["all"]["f1"]),
            "regional_xgb_all_preds" : reg_eval["xgb_all_preds"],
            "regional_xgb_all_acc"   : np.float32(reg_eval["xgb_all_acc"]),
            "regional_xgb_all_f1"    : np.float32(reg_eval["xgb_all_f1"]),
        })

    # only present when RUN_KNN=True 
    if knn_result is not None:
        save_dict.update({
            "knn_preds"              : knn_result["preds"],
            "knn_acc"                : np.float32(knn_result["acc"]),
            "knn_f1"                 : np.float32(knn_result["f1"]),
            "knn_pca_embeddings_test"  : knn_result["pca_embeddings_test"],
            "knn_pca_embeddings_train" : knn_result["pca_embeddings_train"],
            "knn_n_components"         : np.int32(knn_result["n_components"]),
        })

    np.savez_compressed(out_path, **save_dict)
    print(f"  Saved: {out_path}")
    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()

"""
OPERA DISP-S1 Classification: Visualization

Loads results/predictions.npz and saves five figures to figures/:

  1. umap_comparison.png: three-panel UMAP on the test set
       (a) temporal engineered features
       (b) 1D CNN embeddings from the temporal branch
       (c) all engineered features (temporal + spatial/DEM)

  2. confusion_matrices.png: normalized heatmaps for LR, CNN, and XGBoost.

  3. feature_importance.png: top-20 XGBoost gain importances.

  4. ablation_study.png: accuracy and macro-F1 for temporal-only,
     spatial-only, and all-features XGBoost runs.

  5. regional_generalization.png: random-split vs regional hold-out
     macro-F1 across all models.

Usage:
    cd src && python visualize.py
"""

import sys
import types as _types
if "tensorflow" not in sys.modules:
    _tf_mock = _types.ModuleType("tensorflow")
    sys.modules["tensorflow"] = _tf_mock

# Windows terminals default to CP-1252, which breaks Unicode in print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from config import (
    RESULTS_DIR, FIGURES_DIR, PREDICTIONS_NPZ,
    TEMPORAL_KEYWORDS, SPATIAL_PREFIXES,
    LABEL_COLORS, PLOT_STYLE, TITLE_FONT, DPI,
)


# Helper utilities

def _setup_style():
    """Applies a consistent matplotlib/seaborn style for all figures."""
    plt.style.use(PLOT_STYLE)
    sns.set_context("paper", font_scale=1.2)


def _save(fig: plt.Figure, filename: str):
    """Saves a figure to FIGURES_DIR and closes it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _legend_patches(label_classes: np.ndarray) -> list:
    """Creates matplotlib legend patches for the class colour palette."""
    return [
        mpatches.Patch(color=LABEL_COLORS.get(c, "#aaaaaa"), label=c)
        for c in label_classes
    ]


def _class_colors(label_classes: np.ndarray, y_enc: np.ndarray) -> list:
    """Maps encoded integer labels to hex colour strings."""
    return [LABEL_COLORS.get(label_classes[i], "#aaaaaa") for i in y_enc]


# UMAP comparison

def plot_umap_comparison(data: dict):
    """Three-panel UMAP comparison on the test set"""
    print("\n[1/5]  UMAP comparison (3 panels) ...")
    y_test        = data["y_test"]
    label_classes = data["label_classes"]
    emb_test      = data["cnn_embeddings_test"]
    temp_feat     = data["temporal_feat_test"]
    all_feat      = data["all_feat_test"]

    scaler = StandardScaler()
    temp_feat_sc = scaler.fit_transform(temp_feat.astype(np.float64))
    all_feat_sc  = scaler.fit_transform(all_feat.astype(np.float64))

    umap_kwargs = dict(n_components=2, n_neighbors=15, min_dist=0.1,
                       random_state=42, verbose=False)

    print("    Fitting UMAP on temporal engineered features ...")
    umap_temp = UMAP(**umap_kwargs).fit_transform(temp_feat_sc)
    print("    Fitting UMAP on CNN embeddings ...")
    umap_cnn  = UMAP(**umap_kwargs).fit_transform(emb_test.astype(np.float64))
    print("    Fitting UMAP on all engineered features ...")
    umap_all  = UMAP(**umap_kwargs).fit_transform(all_feat_sc)

    colors  = _class_colors(label_classes, y_test)
    patches = _legend_patches(label_classes)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("UMAP Projection — Test Set  (same colour scale, same UMAP params)",
                 **TITLE_FONT)

    panels = [
        (umap_temp, "Temporal Engineered Features\n(17 hand-crafted)"),
        (umap_cnn,  "1D CNN Embeddings\n(learned from raw time series)"),
        (umap_all,  "All Engineered Features\n(temporal + spatial/DEM)"),
    ]
    for ax, (emb, title) in zip(axes, panels):
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=14, alpha=0.7, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("UMAP dim 1", fontsize=9)
        ax.set_ylabel("UMAP dim 2", fontsize=9)
        ax.legend(handles=patches, loc="best", fontsize=8,
                  markerscale=1.5, framealpha=0.85,
                  title="Class", title_fontsize=8)

    plt.tight_layout()
    _save(fig, "umap_comparison.png")


# Confusion matrices

def plot_confusion_matrices(data: dict):
    """Normalized confusion matrix heatmaps for LR, CNN, and XGBoost.

    Row-normalized so per-class recall is easy to read.
    """
    print("\n[2/5]  Confusion matrices ...")
    y_test        = data["y_test"]
    label_classes = data["label_classes"]

    models = [
        (data["lr_preds"],  "Logistic Regression"),
        (data["cnn_preds"], "Hybrid 1D CNN"),
    ]
    if "xgb_all_preds" in data:
        models.append((data["xgb_all_preds"], "XGBoost (All Features)"))

    n_panels = len(models)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — Normalised by True Class", **TITLE_FONT)

    for ax, (preds, title) in zip(axes, models):
        cm = confusion_matrix(y_test, preds, normalize="true")
        sns.heatmap(
            cm,
            annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_classes, yticklabels=label_classes,
            linewidths=0.5, linecolor="white",
            ax=ax, vmin=0, vmax=1,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    _save(fig, "confusion_matrices.png")


# Feature importance

def plot_feature_importance(data: dict, top_n: int = 20):
    """Horizontal bar chart of XGBoost gain importance (combined feature run)."""
    print("\n[3/5]  Feature importance ...")
    if "xgb_feature_names" not in data:
        print("  Skipped — XGBoost data not in predictions.npz (RUN_XGB=False)")
        return
    names      = data["xgb_feature_names"]
    importance = data["xgb_feature_importance"]

    order = np.argsort(importance)[::-1][:top_n]
    top_names = names[order]
    top_vals  = importance[order]

    def _is_temporal(col: str) -> bool:
        if col.startswith(SPATIAL_PREFIXES):
            return False
        return any(kw in col for kw in TEMPORAL_KEYWORDS)

    bar_colors = ["#4363d8" if _is_temporal(n) else "#f58231"
                  for n in top_names]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color=bar_colors[::-1], alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("XGBoost Gain Importance", fontsize=10)
    ax.set_title(f"Top-{top_n} Feature Importances  (XGBoost, All Features)",
                 **TITLE_FONT)

    leg_patches = [
        mpatches.Patch(color="#4363d8", label="Temporal"),
        mpatches.Patch(color="#f58231", label="Spatial / DEM"),
    ]
    ax.legend(handles=leg_patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "feature_importance.png")


# Ablation study

def plot_ablation_study(data: dict):
    """Grouped bar chart of accuracy and macro-F1 for the three ablation conditions."""
    print("\n[4/5]  Ablation study ...")
    if "ablation_temporal_acc" not in data:
        print("  Skipped — XGBoost ablation data not in predictions.npz (RUN_XGB=False)")
        return

    runs = ["temporal", "spatial", "all"]
    run_labels = ["Temporal\nOnly", "Spatial/DEM\nOnly", "All Features"]
    accs = [float(data[f"ablation_{r}_acc"]) for r in runs]
    f1s  = [float(data[f"ablation_{r}_f1"])  for r in runs]

    x     = np.arange(len(runs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_acc = ax.bar(x - width / 2, accs, width,
                      label="Accuracy",  color="#4363d8", alpha=0.85)
    bars_f1  = ax.bar(x + width / 2, f1s,  width,
                      label="Macro F1",  color="#e6194b", alpha=0.85)

    # Annotate bar heights
    for bar_group in (bars_acc, bars_f1):
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("XGBoost Ablation Study:  Temporal vs Spatial/DEM vs All",
                 **TITLE_FONT)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "ablation_study.png")


# Regional generalization

def plot_regional_generalization(data: dict):
    """Grouped bar chart: random-split vs regional hold-out macro-F1 per model."""
    print("\n[5/5]  Regional generalisation ...")

    from sklearn.metrics import f1_score as _f1

    models   = ["LR", "Hybrid CNN"]
    rand_f1s = [
        _f1(data["y_test"], data["lr_preds"],  average="macro"),
        _f1(data["y_test"], data["cnn_preds"], average="macro"),
    ]
    reg_f1s = [
        float(data["regional_lr_f1"]),
        float(data["regional_cnn_f1"]),
    ]

    # XGBoost — only when present
    if "xgb_all_preds" in data:
        models.append("XGBoost\n(All Features)")
        rand_f1s.append(_f1(data["y_test"], data["xgb_all_preds"], average="macro"))
        reg_f1s.append(float(data["regional_xgb_all_f1"]))

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_rand = ax.bar(x - width / 2, rand_f1s, width,
                       label="Random Split (80/20)",
                       color="#4363d8", alpha=0.85)
    bars_reg  = ax.bar(x + width / 2, reg_f1s,  width,
                       label="Regional Hold-out",
                       color="#e6194b", alpha=0.85, hatch="//", edgecolor="white")

    # Value labels
    for bar_group in (bars_rand, bars_reg):
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # Gap annotation arrows between pairs
    for i, (r, g) in enumerate(zip(rand_f1s, reg_f1s)):
        gap = r - g
        if abs(gap) > 0.01:
            mid_x = x[i]
            ax.annotate(f"-{gap:.2f}", xy=(mid_x, min(r, g) - 0.04), ha="center", fontsize=8.5, color="#555555", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro F1", fontsize=10)
    ax.set_title(
        "Geographic Generalisation:  Random Split vs Regional Hold-out",
        **TITLE_FONT,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save(fig, "regional_generalization.png")

# Main

def main():
    """Loads predictions.npz and generates all five figures."""
    print("=" * 68)
    print("OPERA DISP-S1  --  Visualization Pipeline")
    print("=" * 68)

    if not PREDICTIONS_NPZ.exists():
        raise FileNotFoundError(
            f"Results file not found: {PREDICTIONS_NPZ}\n"
            "Run train.py first to generate it."
        )

    print(f"\nLoading {PREDICTIONS_NPZ} ...")
    raw = np.load(PREDICTIONS_NPZ, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    for key in ("label_classes", "xgb_feature_names", "temporal_feat_names", "all_feat_names"):
        if key in data:
            data[key] = data[key].astype(str)

    print(f"  Keys loaded      : {sorted(data.keys())}")
    print(f"  Test samples     : {len(data['y_test'])}")
    print(f"  Label classes    : {list(data['label_classes'])}")
    print(f"  CNN emb shape    : {data['cnn_embeddings_test'].shape}")
    print(f"  Temp feat shape  : {data['temporal_feat_test'].shape}")
    print(f"  All feat shape   : {data['all_feat_test'].shape}")

    _setup_style()

    plot_umap_comparison(data)
    plot_confusion_matrices(data)
    plot_feature_importance(data)
    plot_ablation_study(data)
    plot_regional_generalization(data)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

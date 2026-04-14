"""Shared configuration for the OPERA DISP-S1 classification pipeline"""

from pathlib import Path


PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"
MODELS_DIR    = PROJECT_ROOT / "models"
FIGURES_DIR   = PROJECT_ROOT / "figures"


PREDICTIONS_NPZ = RESULTS_DIR / "predictions.npz"


META_COLS = ["pixel_y", "pixel_x", "frame_id", "region", "label"]


TEMPORAL_KEYWORDS = [
    "rate", "vel", "amplitude", "seasonal", "trend",
    "acceleration", "curvature", "disp", "cumulative",
    "r_squared", "residual", "autocorr", "r2_ratio",
    "monoton", "num_valid", "kurtosis", "skewness",
]


SPATIAL_PREFIXES = ("nbr_", "spatial_")


RANDOM_SEED = 42
TEST_SIZE   = 0.2


HOLDOUT_REGIONS = {
    "tectonic"  : "san_andreas",       
    "subsidence": "san_joaquin",       
    "stable"    : "stable_texas",      
    "landslide" : "california_north",  
}


CNN_SEQ_LEN         = 100
CNN_EPOCHS          = 60
CNN_BATCH_SIZE      = 64
CNN_VAL_SPLIT       = 0.15   # fraction of training data held out for early stopping
EARLY_STOP_PATIENCE = 10


LABEL_COLORS = {
    "tectonic"  : "#e6194b",
    "landslide" : "#3cb44b",
    "subsidence": "#4363d8",
    "stable"    : "#f58231",
}
PLOT_STYLE = "seaborn-v0_8-whitegrid"
TITLE_FONT = {"fontsize": 13, "fontweight": "bold"}
DPI        = 150


REGIONS = {
    # --- Tectonic ---
    "san_andreas": {
        "label": "tectonic",
        "bbox": (-121.0, 35.8, -120.3, 36.5),
        "shapefile": "auxiliary/faults/Qfaults_US_Database.shp",
        "shapefile_filter": {"column": "fault_name", "contains": "San Andreas"},
        "buffer_m": 2000,
    },
    "hayward": {
        "label": "tectonic",
        "bbox": (-122.3, 37.4, -121.8, 37.8),
        "shapefile": "auxiliary/faults/Qfaults_US_Database.shp",
        "shapefile_filter": {"column": "fault_name", "contains": "Hayward"},
        "buffer_m": 2000,
    },
    "wasatch": {
        "label": "tectonic",
        "bbox": (-112.1, 40.5, -111.7, 41.0),
        "shapefile": "auxiliary/faults/Qfaults_US_Database.shp",
        "shapefile_filter": {"column": "fault_name", "contains": "Wasatch"},
        "buffer_m": 2000,
    },
    # --- Landslide ---
    "oregon_coast": {
        "label": "landslide",
        "bbox": (-124.2, 43.5, -123.5, 44.5),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 500,
    },
    "california_north": {
        "label": "landslide",
        "bbox": (-124.5, 40.0, -123.5, 41.5),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 500,
    },
    "washington": {
        "label": "landslide",
        "bbox": (-123.0, 46.5, -122.0, 47.5),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 500,
    },
    # --- Subsidence ---
    "san_joaquin": {
        "label": "subsidence",
        "bbox": (-120.3, 35.5, -119.5, 36.5),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "houston": {
        "label": "subsidence",
        "bbox": (-95.8, 29.5, -95.0, 30.1),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "phoenix": {
        "label": "subsidence",
        "bbox": (-112.3, 33.2, -111.6, 33.7),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    # --- Stable ---
    "stable_sierra": {
        "label": "stable",
        "bbox": (-119.5, 37.0, -118.8, 37.4),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "stable_texas": {
        "label": "stable",
        "bbox": (-98.0, 30.0, -97.2, 30.5),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "stable_kansas": {
        "label": "stable",
        "bbox": (-97.5, 38.5, -96.8, 39.0),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
}

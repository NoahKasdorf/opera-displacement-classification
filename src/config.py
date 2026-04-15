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


CNN_SEQ_LEN         = 200
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
    # --- Landslide ---
    "oregon_coast": {
        "label": "landslide",
        "bbox": (-124.2, 43.5, -123.5, 44.5),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 1000,
    },
    "california_north": {
        "label": "landslide",
        "bbox": (-124.5, 40.0, -123.5, 41.5),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 1000,
    },
    "washington": {
        "label": "landslide",
        "bbox": (-123.0, 46.5, -122.0, 47.5),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 1000,
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
    # --- Tectonic (additions) ---
    "garlock": {
        "label": "tectonic",
        "bbox": (-117.8, 35.1, -117.0, 35.5),
        "shapefile": "auxiliary/faults/Qfaults_US_Database.shp",
        "shapefile_filter": {"column": "fault_name", "contains": "Garlock"},
        "buffer_m": 2000,
    },
    "imperial": {
        "label": "tectonic",
        "bbox": (-115.8, 32.7, -115.2, 33.1),
        "shapefile": "auxiliary/faults/Qfaults_US_Database.shp",
        "shapefile_filter": {"column": "fault_name", "contains": "Imperial"},
        "buffer_m": 2000,
    },
    "san_jacinto": {
        "label": "tectonic",
        "bbox": (-117.0, 33.5, -116.3, 34.0),
        "shapefile": "auxiliary/faults/Qfaults_US_Database.shp",
        "shapefile_filter": {"column": "fault_name", "contains": "San Jacinto"},
        "buffer_m": 2000,
    },
    "owens_valley": {
        "label": "tectonic",
        "bbox": (-118.3, 36.8, -117.6, 37.4),
        "shapefile": "auxiliary/faults/Qfaults_US_Database.shp",
        "shapefile_filter": {"column": "fault_name", "contains": "Owens"},
        "buffer_m": 2000,
    },
    # --- Landslide (additions) ---
    "slumgullion": {
        "label": "landslide",
        # Active lobe of Slumgullion earthflow, SE of Lake City CO (~20 mm/yr)
        "bbox": (-107.32, 37.95, -107.18, 38.08),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": 1000,
    },
    "grand_mesa": {
        "label": "landslide",
        # Shifted east to stay within OPERA frame 14873 (prev bbox hit left edge at col 0)
        # Drop shapefile — USLS point inventory is sparse here; bbox captures escarpment slides
        "bbox": (-108.1, 38.85, -107.7, 39.15),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": 1000,
    },
    "oso_washington": {
        "label": "landslide",
        "bbox": (-121.9, 48.1, -121.3, 48.6),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 1000,
    },
    # Portuguese Bend / Rancho Palos Verdes CA — urban landslide complex,
    # 20–100+ mm/yr, semi-arid coastal bluffs, excellent InSAR coherence
    "portuguese_bend": {
        "label": "landslide",
        "bbox": (-118.42, 33.72, -118.33, 33.78),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 1000,
    },
    # La Conchita CA — coastal bluff, multiple historical failures, active creep
    "la_conchita": {
        "label": "landslide",
        "bbox": (-119.50, 34.34, -119.43, 34.40),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 1000,
    },
    # Thistle UT — semi-arid Utah foothills, ongoing movement in Spanish Fork Canyon
    "thistle_utah": {
        "label": "landslide",
        "bbox": (-111.55, 39.85, -111.35, 40.05),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 1000,
    },
    # --- Subsidence (additions) ---
    "las_vegas": {
        "label": "subsidence",
        "bbox": (-115.4, 36.0, -114.7, 36.5),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "permian_basin": {
        "label": "subsidence",
        "bbox": (-103.0, 31.5, -102.0, 32.3),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "stockton_delta": {
        "label": "subsidence",
        "bbox": (-121.5, 37.8, -121.0, 38.2),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "tucson": {
        "label": "subsidence",
        "bbox": (-111.1, 32.1, -110.7, 32.4),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    # --- Stable (additions) ---
    "stable_nebraska": {
        "label": "stable",
        "bbox": (-99.5, 41.5, -98.5, 42.2),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "stable_iowa": {
        "label": "stable",
        "bbox": (-93.5, 41.8, -92.5, 42.5),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "stable_wyoming": {
        "label": "stable",
        "bbox": (-106.5, 42.5, -105.5, 43.2),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "stable_colorado": {
        "label": "stable",
        "bbox": (-104.5, 38.5, -103.5, 39.2),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
}

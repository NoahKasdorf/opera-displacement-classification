"""Shared configuration for OPERA DISP-S1 pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
        "bbox": (-120.8, 35.5, -119.5, 36.5),
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
    # --- Volcanic ---
    "long_valley": {
        "label": "volcanic",
        "bbox": (-119.2, 37.5, -118.5, 37.9),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "yellowstone": {
        "label": "volcanic",
        "bbox": (-111.0, 44.3, -110.2, 44.8),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    # --- Stable ---
    "stable_sierra": {
        "label": "stable",
        "bbox": (-119.5, 37.0, -118.8, 37.5),
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
}

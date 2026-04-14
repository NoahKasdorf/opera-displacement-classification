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
    "wasatch": {
        "label": "tectonic",
        "bbox": (-112.1, 40.5, -111.7, 41.0),
        "shapefile": "auxiliary/faults/Qfaults_US_Database.shp",
        "shapefile_filter": {"column": "fault_name", "contains": "Wasatch"},
        "buffer_m": 2000,
    },
    # --- Landslide ---
    #"oregon_coast": { #remove
    #    "label": "landslide",
    #    "bbox": (-124.2, 43.5, -123.5, 44.5),
    #    "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
    #    "shapefile_filter": None,
    #    "buffer_m": 500,
    #},
    "california_north": {
        "label": "landslide",
        "bbox": (-124.5, 40.0, -123.5, 41.5),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 500,
    },
    #"washington": { # remove
    #    "label": "landslide",
    #    "bbox": (-121.8, 43.6, -122.8, 44.6),
    #    #"bbox": (-123.0, 46.5, -122.0, 47.5),
    #    "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
    #    "shapefile_filter": None,
    #    "buffer_m": 500,
    #},
    "new_mexico": {
        "label": "landslide",
        "bbox": (-105.8, 36.0, -104.8, 37.0),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 500,
    },
    "colorado": {
        "label": "landslide",
        "bbox": (-105.5, 39.7, -105.2, 40.2),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 500,
    },
    "santa_barbara": {
        "label": "landslide",
        "bbox": (-120.0, 34.4, -119.4, 34.7),
        "shapefile": "auxiliary/landslides/us_ls_v3_point.shp",
        "shapefile_filter": None,
        "buffer_m": 500,
    },
    # --- Subsidence ---
    "san_joaquin": {
        "label": "subsidence",
        # "bbox": (-120.8, 35.5, -119.5, 36.5),
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

    # --- Volcanic ---
    # "long_valley": {
    #     "label": "volcanic",
    #     "bbox": (-119.2, 37.5, -118.5, 37.9),
    #     "shapefile": None,
    #     "shapefile_filter": None,
    #     "buffer_m": None,
    # },
    # "yellowstone": {
    #     "label": "volcanic",
    #     "bbox": (-111.0, 44.3, -110.2, 44.8),
    #     "shapefile": None,
    #     "shapefile_filter": None,
    #     "buffer_m": None,
    # },
    # "st_helens": {
    #     "label": "volcanic",
    #     "bbox": (-122.5, 46.0, -121.8, 46.4),
    #     "shapefile": None,
    #     "shapefile_filter": None,
    #     "buffer_m": None,
    # },
    # --- Stable ---
    "stable_sierra": {
        "label": "stable",
        # "bbox": (-119.5, 37.0, -118.8, 37.5),
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
    "stable_utah": {
        "label": "stable",
        "bbox": (-113.6, 40.6, -113.2, 40.8),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
    "stable_nevada": {  # verify
        "label": "subsidence",
        "bbox": (-115.5, 35.5, -114.6, 36.0),
        "shapefile": None,
        "shapefile_filter": None,
        "buffer_m": None,
    },
}

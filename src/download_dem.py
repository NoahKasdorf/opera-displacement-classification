"""
Download GLO-30 DEM tiles for each OPERA DISP-S1 frame.
Run once before preprocessing.

How it works:
  1. For each region, finds all unique OPERA frame IDs from the .nc filenames
  2. Reads the UTM coordinates from one file per frame, converts to lat/lon
  3. Downloads the Copernicus GLO-30 DEM tiles covering that bounding box
     from a public AWS bucket (no auth needed)
  4. Stitches tiles into one GeoTIFF per frame at data/dem/dem_{frame_id}.tif

Usage:
    python download_dem.py
"""

import re
import rasterio
import numpy as np
import xarray as xr
from pathlib import Path
from dem_stitcher import stitch_dem
from pyproj import CRS, Transformer
from config import PROJECT_ROOT, REGIONS


def get_frame_bounds_latlon(nc_path):
    """Gets lat/lon bounding box from an OPERA file's UTM coordinates."""
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    wkt = ds["spatial_ref"].attrs["crs_wkt"]
    x = ds.x.values
    y = ds.y.values
    ds.close()

    src_crs = CRS.from_wkt(wkt)
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    # Transform the four corner coordinates from UTM to lat/lon
    lons, lats = transformer.transform(
        [x[0], x[-1], x[0], x[-1]],
        [y[0], y[0], y[-1], y[-1]],
    )

    return [min(lons), min(lats), max(lons), max(lats)]


def download_dem_for_region(region_name):
    data_dir = PROJECT_ROOT / "data" / region_name
    dem_dir = PROJECT_ROOT / "data" / "dem"
    dem_dir.mkdir(exist_ok=True)

    nc_files = sorted(data_dir.glob("*.nc"))
    if not nc_files:
        print(f"No .nc files in {data_dir}")
        return

    # Find unique frame IDs, keep one sample file per frame
    frames = {}
    for f in nc_files:
        match = re.search(r"_[Ff](\d+)_", f.name)
        if match:
            fid = match.group(1)
            if fid not in frames:
                frames[fid] = str(f)

    for frame_id, sample_file in frames.items():
        out_path = dem_dir / f"dem_{frame_id}.tif"
        if out_path.exists():
            print(f"  Already have: {out_path}")
            continue

        print(f"  Frame {frame_id}: reading bounds...")
        bounds = get_frame_bounds_latlon(sample_file)
        print(f"  Bounds: {bounds}")

        # Small buffer so DEM fully covers the OPERA frame after reprojection
        buf = 0.05
        bounds = [bounds[0] - buf, bounds[1] - buf, bounds[2] + buf, bounds[3] + buf]

        print(f"  Downloading GLO-30 tiles...")
        dem, profile = stitch_dem(
            bounds,
            dem_name="glo_30",
            dst_ellipsoidal_height=False,
        )

        # profile contains everything rasterio needs: CRS, transform, compression, etc.
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(dem, 1)

        print(f"  Saved: {out_path} ({dem.shape})")


if __name__ == "__main__":
    for region_name in REGIONS:
        print(f"\n{'='*40}")
        print(f"DEM for: {region_name}")
        download_dem_for_region(region_name)

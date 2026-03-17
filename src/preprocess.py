"""
OPERA DISP-S1 Preprocessing Pipeline
=====================================
Groups granules by frame ID, processes each frame independently,
merges at the feature level.

Usage:
    python preprocess.py --region san_andreas
    python preprocess.py --combine
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import re
import argparse
import pandas as pd
from scipy import stats
from datetime import datetime
from pathlib import Path
from pyproj import CRS
from shapely.geometry import Point, box

from config import PROJECT_ROOT, REGIONS


def parse_filename(filepath):
    """Extracts the OPERA frame ID and acquisition date from a file path.

    Args:
        filepath (str or Path): The full path or filename to parse.

    Returns:
        tuple: A (frame_id (str), acq_date (datetime)) pair.
    """
    basename = Path(filepath).name

    # Fixed: Regex modified to handle any number of digits; failure raises an error.
    frame_match = re.search(r"_[Ff](\d+)_", basename)
    if not frame_match:
        raise ValueError(f"Can't parse frame ID from: {basename}")
    frame_id = frame_match.group(1)

    date_matches = re.findall(r"(\d{8})T\d{6}Z", basename)
    if len(date_matches) >= 2:
        acq_date = datetime.strptime(date_matches[1], "%Y%m%d")
    elif date_matches:
        acq_date = datetime.strptime(date_matches[0], "%Y%m%d")
    else:
        raise ValueError(f"Can't parse date from: {basename}")

    return frame_id, acq_date


def group_files_by_frame(data_dir):
    """Groups netCDF files by frame ID and deduplicates identical dates.

    Args:
        data_dir (str or Path): Directory containing the .nc files.

    Returns:
        dict: A dictionary mapping frame IDs to sorted lists of (date, path) tuples.
    """
    nc_files = sorted(Path(data_dir).glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files in {data_dir}")

    groups = {}
    for f in nc_files:
        try:
            frame_id, acq_date = parse_filename(str(f))
            groups.setdefault(frame_id, []).append((acq_date, str(f)))
        except ValueError as e:
            print(f"  Skipping: {e}")

    # Fixed: Deduplicate by secondary date, keeping the one with the latest creation date
    for fid in groups:
        seen = {}
        for date, path in groups[fid]:
            seen[date] = (date, path)  # Last one overwrites previous identical dates
        groups[fid] = sorted(seen.values(), key=lambda x: x[0])

    print(
        f"  Found {len(nc_files)} files across {len(groups)} frame(s): "
        f"{', '.join(f'{k}({len(v)})' for k, v in groups.items())}"
    )
    return groups


def load_frame_datasets(file_list):
    """Loads and validates a time series of datasets for a single frame.

    Args:
        file_list (list): List of (date, path) tuples.

    Returns:
        tuple: A list of datetime objects and a list of open xarray Datasets.
    """
    dates = [d for d, _ in file_list]
    datasets = [xr.open_dataset(p, engine="h5netcdf") for _, p in file_list]

    # Fixed: Validate required variables to prevent cryptic crashes
    REQUIRED_VARS = {
        "displacement",
        "recommended_mask",
        "temporal_coherence",
        "spatial_ref",
    }
    for ds, (_, path) in zip(datasets, file_list):
        missing = REQUIRED_VARS - set(ds.data_vars) - set(ds.coords)
        if missing:
            raise ValueError(f"Missing variables in {path}: {missing}")

    # Fixed: Halt execution if there are multiple reference epochs
    ref_dates = {str(ds.attrs.get("reference_datetime", "?")) for ds in datasets}
    if len(ref_dates) > 1:
        raise ValueError(
            f"Multiple reference epochs detected: {ref_dates}. Cannot process continuous time series."
        )

    print(
        f"  Loaded {len(datasets)} timesteps: {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}"
    )
    print(f"  Grid shape: {datasets[0]['short_wavelength_displacement'].shape}")
    return dates, datasets


def get_quality_mask(ds):
    """Generates a boolean mask keeping only high-quality, high-coherence pixels.

    Args:
        ds (xarray.Dataset): A single OPERA displacement dataset.

    Returns:
        tuple: A boolean quality mask and the temporal coherence array.
    """
    rec = ds["recommended_mask"].values == 1
    coherence = ds["temporal_coherence"].values

    # Fixed: Removed explicit water mask check. `recommended_mask` already excludes water.
    good = rec & (coherence > 0.5)
    print(
        f"  Quality: {np.sum(good):,} / {good.size:,} pass "
        f"({100 * np.sum(good) / good.size:.1f}%)"
    )
    return good, coherence


def sample_pixels(good_mask, min_spacing_m=500, pixel_size_m=30):
    """Downsamples valid pixels to a regular grid based on minimum spacing.

    Args:
        good_mask (numpy.ndarray): Boolean mask of valid pixels.
        min_spacing_m (int, optional): Minimum distance between samples. Defaults to 500.
        pixel_size_m (int, optional): Native pixel resolution. Defaults to 30.

    Returns:
        tuple: Arrays of (y_coords, x_coords) for the sampled pixels.
    """
    step = max(1, int(min_spacing_m / pixel_size_m))
    ys, xs = np.where(good_mask)
    on_grid = (ys % step == 0) & (xs % step == 0)
    ys, xs = ys[on_grid], xs[on_grid]
    print(f"  Sampled {len(ys):,} pixels (spacing {min_spacing_m}m = {step}px)")
    return ys, xs


def get_opera_crs(ds):
    """Extracts the coordinate reference system from the dataset attributes.

    Args:
        ds (xarray.Dataset): The OPERA dataset.

    Returns:
        pyproj.CRS: The parsed coordinate reference system.
    """
    wkt = ds["spatial_ref"].attrs.get("crs_wkt")
    # Fixed: Raise error instead of silently falling back to a hardcoded UTM zone
    if not wkt:
        raise ValueError("No crs_wkt in spatial_ref attrs — cannot determine CRS")
    return CRS.from_wkt(wkt)


def filter_by_shapefile(y_coords, x_coords, ds, region_info):
    """Filters pixel coordinates to keep only those inside a designated hazard zone.

    Args:
        y_coords (numpy.ndarray): Y coordinates (row indices).
        x_coords (numpy.ndarray): X coordinates (column indices).
        ds (xarray.Dataset): The reference dataset for spatial coordinates.
        region_info (dict): Configuration dictionary containing shapefile paths and filters.

    Returns:
        tuple: Filtered (y_coords, x_coords) arrays.
    """
    shp_path = PROJECT_ROOT / region_info["shapefile"]
    if not shp_path.exists():
        print(f"  WARNING: Shapefile not found: {shp_path}")
        return y_coords, x_coords

    gdf = gpd.read_file(shp_path)

    # Attribute filter
    sf = region_info.get("shapefile_filter")
    if sf:
        gdf = gdf[gdf[sf["column"]].str.contains(sf["contains"], case=False, na=False)]
        print(f"  Attribute filter: {len(gdf)} features matching '{sf['contains']}'")

    # Spatial clip to region bbox (Fixed: Using a single robust reprojection flow to avoid index misalignment)
    bbox = region_info["bbox"]
    bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.intersects(bbox_poly)]
    print(f"  Bbox clip: {len(gdf)} features")

    if len(gdf) == 0:
        print(f"  WARNING: No features after filtering!")
        return np.array([]), np.array([])

    # Reproject to OPERA CRS and buffer (Fixed: passing pyproj.CRS directly)
    opera_crs = get_opera_crs(ds)
    gdf = gdf.to_crs(opera_crs)

    buffer_m = region_info.get("buffer_m")
   # zone = gdf.buffer(buffer_m).unary_union if buffer_m else gdf.geometry.unary_union
    zone = gdf.buffer(buffer_m).union_all() if buffer_m else gdf.geometry.union_all()

    # Point-in-polygon
    x_utm, y_utm = ds.x.values, ds.y.values
    points = gpd.GeoSeries(
        [Point(x_utm[x], y_utm[y]) for y, x in zip(y_coords, x_coords)]
    )
    inside = points.within(zone)
    print(f"  Hazard zone: {inside.sum()} / {len(y_coords)} pixels inside")
    return y_coords[inside.values], x_coords[inside.values]


def filter_by_rate(y_coords, x_coords, cube, t_days, mode, threshold_mm_yr):
    """Filters pixels based on their linear displacement rate to isolate stable or subsiding areas.

    Args:
        y_coords (numpy.ndarray): Y coordinates to evaluate.
        x_coords (numpy.ndarray): X coordinates to evaluate.
        cube (numpy.ndarray): 3D array of displacement data.
        t_days (numpy.ndarray): 1D array of time days from the reference epoch.
        mode (str): Filtering mode ('stable' or 'subsidence').
        threshold_mm_yr (float): Threshold rate in millimeters per year.

    Returns:
        tuple: Filtered (y_coords, x_coords) arrays.
    """
    # Fixed: Vectorized approach replacing the slow per-pixel loop
    ts_matrix = cube[:, y_coords, x_coords]  # (T, N)
    valid = ~np.isnan(ts_matrix)

   #t_mean = np.average(t_days[:, None], weights=valid, axis=0)
   
    valid_counts = np.sum(valid, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mean = np.sum(t_days[:, None] * valid, axis=0) / valid_counts
   
    y_mean = np.nanmean(ts_matrix, axis=0)

    t_diff = t_days[:, None] - t_mean
    y_diff = ts_matrix - y_mean

    numerator = np.nansum(t_diff * y_diff, axis=0)
    denominator = np.nansum(valid * (t_diff**2), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_m_day = np.where(denominator != 0, numerator / denominator, np.nan)

    rate_mm_yr = rate_m_day * 365.25 * 1000
    valid_counts = np.sum(valid, axis=0)
    has_data = valid_counts >= 10

    if mode == "stable":
        keep_mask = has_data & (np.abs(rate_mm_yr) <= threshold_mm_yr)
    else:  # subsidence
        keep_mask = has_data & (rate_mm_yr <= -threshold_mm_yr)

    kept_y = y_coords[keep_mask]
    kept_x = x_coords[keep_mask]

    print(
        f"  Rate filter ({mode}, {threshold_mm_yr} mm/yr): {len(kept_y)} / {len(y_coords)}"
    )
    return kept_y, kept_x


def build_displacement_cube(datasets):
    """Stacks short-wavelength displacement layers into a continuous 3D NumPy array.

    Args:
        datasets (list): List of xarray Datasets ordered by time.

    Returns:
        numpy.ndarray: A 3D array of shape (time, y, x).
    """
    # Fixed: Explicit squeeze() added to prevent stacking into a 4D array.
    # Fixed: Switching to short_wavelength_displacement to reduce noise per audit.
    return np.array(
        [ds["short_wavelength_displacement"].values.squeeze() for ds in datasets],
        dtype=float,
    )


def extract_time_series(cube, y_coords, x_coords, dates, max_nan_frac=0.2):
    """Extracts and interpolates 1D displacement time series for the target pixels.

    Args:
        cube (numpy.ndarray): 3D displacement array.
        y_coords (numpy.ndarray): Y coordinates of target pixels.
        x_coords (numpy.ndarray): X coordinates of target pixels.
        dates (list): List of acquisition datetimes.
        max_nan_frac (float, optional): Maximum allowed fraction of missing data. Defaults to 0.2.

    Returns:
        list: Dictionaries containing valid pixel coordinates, interpolated time series, and time arrays.
    """
    T = cube.shape[0]
    t_days = np.array([(d - dates[0]).days for d in dates], dtype=float)

    print(f"  Extracting: {len(y_coords)} pixels x {T} timesteps...")
    valid = []
    skipped = 0

    for i in range(len(y_coords)):
        y, x = int(y_coords[i]), int(x_coords[i])
        ts = cube[:, y, x].copy()

        if np.sum(np.isnan(ts)) / T > max_nan_frac:
            skipped += 1
            continue

        if np.any(np.isnan(ts)):
            good = ~np.isnan(ts)
            ts = np.interp(t_days, t_days[good], ts[good])

        valid.append({"y": y, "x": x, "time_series": ts, "t_days": t_days})

    print(f"  Valid: {len(valid)}, Skipped: {skipped}")
    return valid


def compute_temporal_features(d, t):
    """Calculates temporal statistical features from a single 1D displacement time series.

    Args:
        d (numpy.ndarray): Displacement values.
        t (numpy.ndarray): Time values in days.

    Returns:
        dict or None: Computed features (rate, acceleration, seasonality, etc.), or None if data is insufficient.
    """
    if len(d) < 10:
        return None

    f = {}

    lr = stats.linregress(t, d)
    f["linear_rate_mm_yr"] = lr.slope * 365.25 * 1000
    f["r_squared_linear"] = lr.rvalue**2
    f["total_disp_mm"] = (d[-1] - d[0]) * 1000
    f["max_abs_disp_mm"] = np.max(np.abs(d)) * 1000

    coeffs = np.polyfit(t, d, 2)
    f["acceleration"] = coeffs[0]
    quad_pred = np.polyval(coeffs, t)
    ss_res = np.sum((d - quad_pred) ** 2)
    ss_tot = np.sum((d - np.mean(d)) ** 2)
    r2_quad = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    f["r2_ratio_quad_lin"] = r2_quad / (lr.rvalue**2 + 1e-10)

    omega = 2 * np.pi / 365.25
    A = np.column_stack([t, np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])
    try:
        coefs = np.linalg.lstsq(A, d, rcond=None)[0]
        f["seasonal_amp_mm"] = np.sqrt(coefs[2] ** 2 + coefs[3] ** 2) * 1000
    except np.linalg.LinAlgError:
        f["seasonal_amp_mm"] = 0.0

    residuals = d - (lr.slope * t + lr.intercept)
    f["residual_std_mm"] = np.std(residuals) * 1000
    f["autocorr_lag1"] = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if len(d) > 2 else 0.0

    vel = np.diff(d)
    f["vel_sign_changes"] = int(np.sum(np.diff(np.sign(vel)) != 0))
    if len(vel) > 3:
        f["vel_kurtosis"] = float(stats.kurtosis(vel))
        f["vel_skewness"] = float(stats.skew(vel))
    else:
        f["vel_kurtosis"] = 0.0
        f["vel_skewness"] = 0.0

    return f


def compute_rate_map(cube, t_days, stride=5):
    """Calculates a robust 2D rate map using a stride to save memory."""
    # Only look at every 'stride' pixel (e.g., every 5th pixel)
    # This reduces memory usage by stride^2 (25x for stride=5)
    cube_small = cube[:, ::stride, ::stride]

    valid = ~np.isnan(cube_small)
    valid_counts = np.sum(valid, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_mean = np.sum(t_days[:, None, None] * valid, axis=0) / valid_counts

    y_mean = np.nanmean(cube_small, axis=0)
    t_diff = t_days[:, None, None] - t_mean
    y_diff = cube_small - y_mean

    numerator = np.nansum(t_diff * y_diff, axis=0)
    denominator = np.nansum(valid * (t_diff**2), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(denominator != 0, numerator / denominator, np.nan)

    return rate



def compute_spatial_features(y, x, coherence, rate_map, shape, stride=5):
    """Calculates spatial features using a downsampled rate map."""
    H, W = shape
    f = {"mean_coherence": float(coherence[y, x])}

    # Map the high-res coordinates (y, x) to the low-res rate_map (y_s, x_s)
    y_s, x_s = y // stride, x // stride
    H_s, W_s = rate_map.shape

    # Adjust the radius: if we sample every 5th pixel,
    # a 510m radius (~17px) becomes ~3.4px in the small map.
    r_s = max(1, 17 // stride)

    y0, y1 = max(0, y_s - r_s), min(H_s, y_s + r_s + 1)
    x0, x1 = max(0, x_s - r_s), min(W_s, x_s + r_s + 1)

    nbr = rate_map[y0:y1, x0:x1]
    v = nbr[~np.isnan(nbr)]

    if len(v) > 0:
        f["nbr_mean_rate_mm_yr"] = float(np.mean(v)) * 365.25 * 1000
        f["nbr_std_rate_mm_yr"] = float(np.std(v)) * 365.25 * 1000
    else:
        f["nbr_mean_rate_mm_yr"] = 0.0
        f["nbr_std_rate_mm_yr"] = 0.0

    # Gradient check on the smaller map
    if 1 <= y_s < H_s - 1 and 1 <= x_s < W_s - 1:
        gy = rate_map[y_s + 1, x_s] - rate_map[y_s - 1, x_s]
        gx = rate_map[y_s, x_s + 1] - rate_map[y_s, x_s - 1]
        f["spatial_gradient"] = (
            float(np.sqrt(gy**2 + gx**2)) if not (np.isnan(gy) or np.isnan(gx)) else 0.0
        )
    else:
        f["spatial_gradient"] = 0.0

    return f


def build_features(
    valid_pixels, coherence, rate_map, shape, label, region_name, frame_id, stride=5
):
    """Compiles temporal and spatial features for all valid pixels into a tabular format.

    Args:
        valid_pixels (list): List of pixel dictionaries containing time series data.
        coherence (numpy.ndarray): 2D temporal coherence map.
        rate_map (numpy.ndarray): 2D map of displacement rates.
        shape (tuple): The (height, width) of the spatial grid.
        label (str): Classification label (e.g., 'subsidence').
        region_name (str): Name of the geographical region.
        frame_id (str): OPERA frame identifier.

    Returns:
        pandas.DataFrame: A formatted table of all extracted features.
    """
    print(f"  Computing features for {len(valid_pixels)} pixels...")
    rows = []
    for i, px in enumerate(valid_pixels):
        tf = compute_temporal_features(px["time_series"], px["t_days"])
        if tf is None:
            continue
        sf = compute_spatial_features(px["y"], px["x"], coherence, rate_map, shape, stride=stride)
        row = {
            "pixel_y": px["y"],
            "pixel_x": px["x"],
            "frame_id": frame_id,
            "region": region_name,
            "label": label,
        }
        row.update(tf)
        row.update(sf)
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(valid_pixels)}")

    df = pd.DataFrame(rows)
    print(f"  Result: {len(df)} rows x {len(df.columns)} columns")
    return df


def save_outputs(df, valid_pixels, label, region_name):
    """Writes the feature DataFrame to CSV and saves raw time series to an NPZ archive.

    Args:
        df (pandas.DataFrame): The extracted feature table.
        valid_pixels (list): List of valid pixel dictionaries containing raw time series.
        label (str): Classification label for the data.
        region_name (str): Name of the geographical region.
    """
    out_dir = PROJECT_ROOT / "processed"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / f"features_{region_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(df)} rows)")

    if valid_pixels:
        ts_array = np.array([p["time_series"] for p in valid_pixels])
        npz_path = out_dir / f"timeseries_{region_name}.npz"
        np.savez(
            npz_path,
            time_series=ts_array,
            labels=np.array([label] * len(valid_pixels)),
            regions=np.array([region_name] * len(valid_pixels)),
            t_days=valid_pixels[0]["t_days"],
        )
        print(f"  Saved: {npz_path} (shape {ts_array.shape})")


def process_frame(frame_id, file_list, info, region_name, max_samples):
    """Executes the full preprocessing pipeline on a single stack of frame datasets.

    Args:
        frame_id (str): OPERA frame identifier.
        file_list (list): List of (date, path) tuples for this frame.
        info (dict): Region configuration dictionary.
        region_name (str): Name of the geographical region.
        max_samples (int): Maximum number of pixels to sample.

    Returns:
        tuple: A (DataFrame, list of pixel data) pair, or (None, []) if no data passes filters.
    """
    print(f"\n  --- Frame {frame_id} ({len(file_list)} files) ---")
    dates, datasets = load_frame_datasets(file_list)

    try:
        good_mask, coherence = get_quality_mask(datasets[0])

        # Filter
        label = info["label"]

        if label in ["landslide"]:
            spacing = 90  # 3-pixel spacing
        else:
            spacing = 500  # 16-pixel spacing (Houston/Volcanic/Stable)

        y_coords, x_coords = sample_pixels(good_mask, min_spacing_m=spacing)

        cube = build_displacement_cube(datasets)
        t_days = np.array([(d - dates[0]).days for d in dates], dtype=float)

        if label == "stable":
            y_coords, x_coords = filter_by_rate(
                y_coords, x_coords, cube, t_days, "stable", 2.0
            )
        elif label == "subsidence" and not info.get("shapefile"):
            y_coords, x_coords = filter_by_rate(
                y_coords, x_coords, cube, t_days, "subsidence", 10.0
            )
        elif info.get("shapefile"):
            y_coords, x_coords = filter_by_shapefile(
                y_coords, x_coords, datasets[0], info
            )

        if len(y_coords) == 0:
            print(f"  No pixels remain after filtering.")
            return None, []

        if max_samples and len(y_coords) > max_samples:
            idx = np.random.choice(len(y_coords), max_samples, replace=False)
            y_coords, x_coords = y_coords[idx], x_coords[idx]
            print(f"  Subsampled to {max_samples}")

        valid_pixels = extract_time_series(cube, y_coords, x_coords, dates)
        if not valid_pixels:
            print(f"  No valid time series.")
            return None, []

        stride = 5
        rate_map = compute_rate_map(cube, t_days, stride=stride)
        df = build_features(
            valid_pixels,
            coherence,
            rate_map,
            cube.shape[1:],
            label,
            region_name,
            frame_id,
        )
        return df, valid_pixels

    finally:
        for ds in datasets:
            ds.close()


def process_region(region_name, max_files=None, max_samples=500):
    """Iterates through all frames in a region, processes them, and saves combined outputs.

    Args:
        region_name (str): The region to process as defined in config.py.
        max_files (int, optional): Maximum files to process per frame for testing. Defaults to None.
        max_samples (int, optional): Maximum pixels to sample per frame. Defaults to 500.

    Returns:
        pandas.DataFrame or None: The combined feature table for the entire region.
    """
    info = REGIONS.get(region_name)
    if not info:
        print(f"Unknown region: {region_name}")
        return None

    data_dir = PROJECT_ROOT / "data" / region_name
    print(f"\n{'='*60}")
    print(f"PROCESSING: {region_name} (label: {info['label']})")
    print(f"{'='*60}")

    if not data_dir.exists():
        print(f"  ERROR: No data at {data_dir}")
        return None

    frame_groups = group_files_by_frame(str(data_dir))
    all_dfs, all_pixels = [], []

    for frame_id, file_list in frame_groups.items():
        if max_files:
            file_list = file_list[:max_files]
        df, pixels = process_frame(frame_id, file_list, info, region_name, max_samples)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            all_pixels.extend(pixels)

    if not all_dfs:
        print(f"  ERROR: No data produced for any frame.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    save_outputs(combined_df, all_pixels, info["label"], region_name)

    feat_cols = [
        c
        for c in combined_df.columns
        if c not in ("pixel_y", "pixel_x", "frame_id", "region", "label")
    ]
    print(f"\n  --- Summary ---")
    print(
        f"  Frames: {len(all_dfs)}, Samples: {len(combined_df)}, Features: {len(feat_cols)}"
    )
    for col in feat_cols:
        print(
            f"    {col}: {combined_df[col].min():.4f} to {combined_df[col].max():.4f}"
        )

    return combined_df


def combine_all():
    """Merges all available regional CSV feature files into a single master training dataset.

    Returns:
        pandas.DataFrame or None: The combined master dataset.
    """
    proc_dir = PROJECT_ROOT / "processed"

    csvs = sorted(proc_dir.glob("features_*.csv"))
    csvs = [f for f in csvs if f.name != "features_all.csv"]
    if not csvs:
        print("No CSV files found!")
        return None

    print(f"\nCombining {len(csvs)} files...")
    dfs = [pd.read_csv(f) for f in csvs]
    for f, df in zip(csvs, dfs):
        print(f"  {f.name}: {len(df)} samples, label={df['label'].iloc[0]}")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(proc_dir / "features_all.csv", index=False)

    print(f"\nTotal: {len(combined)} samples")
    print(f"\nClasses:\n{combined['label'].value_counts().to_string()}")
    print(f"\nRegions:\n{combined['region'].value_counts().to_string()}")

    # Fixed: Removed the bug-prone NPZ concatenation step entirely,
    # relying purely on the CSV files for subsequent feature training.

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPERA DISP-S1 Preprocessing")
    parser.add_argument("--region", type=str, help="Region to process")
    parser.add_argument("--combine", action="store_true", help="Combine all regions")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    if args.combine:
        combine_all()
    elif args.region:
        process_region(args.region, args.max_files, args.max_samples)
    else:
        print(
            "Usage:\n  python preprocess.py --region san_andreas\n  python preprocess.py --combine"
        )

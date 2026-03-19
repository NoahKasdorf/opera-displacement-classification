"""
OPERA DISP-S1 Preprocessing Pipeline
=====================================
Extracts temporal, spatial, and terrain features from OPERA DISP-S1
displacement time series for multi-class ground deformation classification.

Pipeline per frame:
  1. Load all granules, read reference epochs from /identification metadata
  2. Crop the full frame grid to the region bounding box (RAM optimization)
  3. Stitch displacement segments across reference epoch changes (per-pixel offsets)
  4. Apply quality mask (recommended_mask + coherence > 0.5)
  5. Filter pixels spatially (shapefile or bbox) then by rate (stable/subsidence)
  6. Extract per-pixel time series and compute features
  7. Add terrain features (slope, aspect) from a co-registered GLO-30 DEM

Usage:
    python preprocess.py --region san_andreas --max-samples 1000
    python preprocess.py --combine
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import re
import argparse
import pandas as pd
from collections import Counter
from scipy import stats
from datetime import datetime
from pathlib import Path
from pyproj import CRS, Transformer
from shapely.geometry import Point, box

import h5py
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

from config import PROJECT_ROOT, REGIONS


def parse_filename(filepath):
    """Extracts OPERA frame ID and secondary acquisition date from a granule filename.

    Filename format:
        OPERA_L3_DISP-S1_IW_F{frame}_{pol}_{ref_date}_{sec_date}_{ver}_{created}.nc

    Returns:
        tuple: (frame_id: str, acq_date: datetime)
    """
    basename = Path(filepath).name

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
    """Groups .nc files by frame ID and deduplicates by secondary date.

    When multiple files share the same secondary date (reprocessed granules),
    keeps the one with the latest creation date (last in lexicographic sort).

    Returns:
        dict: {frame_id: [(date, path), ...]} sorted chronologically.
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

    for fid in groups:
        seen = {}
        for date, path in groups[fid]:
            seen[date] = (date, path)
        groups[fid] = sorted(seen.values(), key=lambda x: x[0])

    print(
        f"  Found {len(nc_files)} files across {len(groups)} frame(s): "
        f"{', '.join(f'{k}({len(v)})' for k, v in groups.items())}"
    )
    return groups


def get_reference_date(filepath):
    """Reads the displacement reference epoch from the HDF5 /identification group."""
    with h5py.File(filepath, "r") as f:
        return f["identification"]["reference_datetime"][()].decode().strip()


def load_frame_datasets(file_list):
    """Loads all granules for a frame sorted by acquisition date.

    Reads reference epoch metadata from each file via h5py (not root attrs,
    which don't contain this field). Returns all files across all reference
    epochs so that build_displacement_cube can stitch them together.

    Returns:
        tuple: (dates, datasets, ref_dates) — parallel lists of acquisition
               datetimes, open xarray Datasets, and reference epoch strings.
    """
    ref_info = []
    for date, path in file_list:
        try:
            ref_date = get_reference_date(path)
            ref_info.append({"ref": ref_date, "acq": date, "path": path})
        except Exception as e:
            print(f"  Skipping {Path(path).name}: {e}")

    if not ref_info:
        raise ValueError("No valid files found.")

    ref_info.sort(key=lambda x: x["acq"])

    dates = [r["acq"] for r in ref_info]
    ref_dates = [r["ref"] for r in ref_info]
    datasets = [xr.open_dataset(r["path"], engine="h5netcdf") for r in ref_info]

    REQUIRED_VARS = {
        "short_wavelength_displacement",
        "recommended_mask",
        "temporal_coherence",
        "spatial_ref",
    }
    for ds, info in zip(datasets, ref_info):
        missing = REQUIRED_VARS - set(ds.data_vars) - set(ds.coords)
        if missing:
            for d in datasets:
                d.close()
            raise ValueError(
                f"Missing variables in {Path(info['path']).name}: {missing}"
            )

    unique_refs = list(dict.fromkeys(ref_dates))
    ref_counts = Counter(ref_dates)

    print(
        f"  Loaded {len(datasets)} timesteps: {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}"
    )
    print(f"  Reference epochs: {len(unique_refs)}")
    for ref in unique_refs:
        print(f"    {ref[:19]}: {ref_counts[ref]} files")
    print(f"  Grid shape: {datasets[0]['short_wavelength_displacement'].shape}")

    return dates, datasets, ref_dates


def get_opera_crs(ds):
    """Reads the CRS from spatial_ref.attrs['crs_wkt'].

    The spatial_ref variable is a scalar int — the CRS lives entirely in
    its attributes, not in the variable value itself.
    """
    wkt = ds["spatial_ref"].attrs.get("crs_wkt")
    if not wkt:
        raise ValueError("No crs_wkt in spatial_ref attrs — cannot determine CRS")
    return CRS.from_wkt(wkt)


def get_bbox_slices(ds, bbox):
    """Converts a (west, south, east, north) lat/lon bbox to row/column slices.

    Used to crop the full OPERA frame grid (~8000x9600) to only the region
    of interest before building the displacement cube. This reduces the cube
    from ~30 GB to under 1 GB, enabling processing on 16 GB RAM machines.

    Returns:
        tuple: (row_slice, col_slice) or None if bbox doesn't intersect the frame.
    """
    opera_crs = get_opera_crs(ds)
    transformer = Transformer.from_crs("EPSG:4326", opera_crs, always_xy=True)

    x_min, y_min = transformer.transform(bbox[0], bbox[1])
    x_max, y_max = transformer.transform(bbox[2], bbox[3])

    x = ds.x.values
    y = ds.y.values

    col_mask = (x >= x_min) & (x <= x_max)
    row_mask = (y >= min(y_min, y_max)) & (y <= max(y_min, y_max))

    cols = np.where(col_mask)[0]
    rows = np.where(row_mask)[0]

    if len(cols) == 0 or len(rows) == 0:
        return None

    row_slice = slice(rows[0], rows[-1] + 1)
    col_slice = slice(cols[0], cols[-1] + 1)

    print(
        f"  Bbox crop: rows {rows[0]}:{rows[-1]+1}, cols {cols[0]}:{cols[-1]+1} "
        f"({len(rows)} × {len(cols)} = {len(rows)*len(cols):,} pixels)"
    )
    return row_slice, col_slice


def get_quality_mask(ds, row_slice=None, col_slice=None):
    """Generates a boolean mask from recommended_mask and temporal coherence.

    The recommended_mask already excludes water pixels, disconnected components,
    and pixels with low coherence + low phase similarity. We add a coherence > 0.5
    threshold on top for additional filtering.

    Returns:
        tuple: (good_mask, coherence) — boolean array and raw coherence values.
    """
    rec = ds["recommended_mask"].values.squeeze()
    coherence = ds["temporal_coherence"].values.squeeze()

    if row_slice and col_slice:
        rec = rec[row_slice, col_slice]
        coherence = coherence[row_slice, col_slice]

    good = (rec == 1) & (coherence > 0.5)
    print(
        f"  Quality: {np.sum(good):,} / {good.size:,} pass "
        f"({100 * np.sum(good) / good.size:.1f}%)"
    )
    return good, coherence


def sample_pixels(good_mask, min_spacing_m=500, pixel_size_m=30):
    """Selects valid pixels on a regular grid with minimum spacing.

    Returns:
        tuple: (y_coords, x_coords) — pixel indices relative to the input mask.
    """
    step = max(1, int(min_spacing_m / pixel_size_m))
    ys, xs = np.where(good_mask)
    on_grid = (ys % step == 0) & (xs % step == 0)
    ys, xs = ys[on_grid], xs[on_grid]
    print(f"  Sampled {len(ys):,} pixels (spacing {min_spacing_m}m = {step}px)")
    return ys, xs


def filter_by_shapefile(y_coords, x_coords, ds, region_info, x_utm=None, y_utm=None):
    """Keeps pixels inside a buffered shapefile zone (faults, landslide points).

    Loads the shapefile, optionally filters by attribute (e.g., fault_name),
    clips to the region bbox, reprojects to the OPERA UTM CRS, buffers, and
    tests point-in-polygon for each pixel coordinate.

    Args:
        x_utm, y_utm: Pre-cropped UTM coordinate arrays. If None, reads from ds.

    Returns:
        tuple: Filtered (y_coords, x_coords).
    """
    shp_path = PROJECT_ROOT / region_info["shapefile"]
    if not shp_path.exists():
        print(f"  WARNING: Shapefile not found: {shp_path}")
        return y_coords, x_coords

    gdf = gpd.read_file(shp_path)

    sf = region_info.get("shapefile_filter")
    if sf:
        gdf = gdf[gdf[sf["column"]].str.contains(sf["contains"], case=False, na=False)]
        print(f"  Attribute filter: {len(gdf)} features matching '{sf['contains']}'")

    bbox = region_info["bbox"]
    bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.intersects(bbox_poly)]
    print(f"  Bbox clip: {len(gdf)} features")

    if len(gdf) == 0:
        print(f"  WARNING: No features after filtering!")
        return np.array([]), np.array([])

    opera_crs = get_opera_crs(ds)
    gdf = gdf.to_crs(opera_crs)

    buffer_m = region_info.get("buffer_m")
    zone = gdf.buffer(buffer_m).union_all() if buffer_m else gdf.geometry.union_all()

    if x_utm is None:
        x_utm = ds.x.values
        y_utm = ds.y.values

    points = gpd.GeoSeries(
        [Point(x_utm[x], y_utm[y]) for y, x in zip(y_coords, x_coords)]
    )
    inside = points.within(zone)
    print(f"  Hazard zone: {inside.sum()} / {len(y_coords)} pixels inside")
    return y_coords[inside.values], x_coords[inside.values]


def filter_by_bbox(y_coords, x_coords, ds, bbox, x_utm=None, y_utm=None):
    """Keeps pixels within a (west, south, east, north) lat/lon bounding box.

    Converts bbox to UTM and tests pixel coordinates against the bounds.
    Used for subsidence, volcanic, and stable regions that lack shapefiles.

    Args:
        x_utm, y_utm: Pre-cropped UTM coordinate arrays. If None, reads from ds.

    Returns:
        tuple: Filtered (y_coords, x_coords).
    """
    opera_crs = get_opera_crs(ds)
    transformer = Transformer.from_crs("EPSG:4326", opera_crs, always_xy=True)

    x_min, y_min = transformer.transform(bbox[0], bbox[1])
    x_max, y_max = transformer.transform(bbox[2], bbox[3])

    if x_utm is None:
        x_utm = ds.x.values
        y_utm = ds.y.values

    x_vals = x_utm[x_coords]
    y_vals = y_utm[y_coords]

    inside = (
        (x_vals >= x_min)
        & (x_vals <= x_max)
        & (y_vals >= min(y_min, y_max))
        & (y_vals <= max(y_min, y_max))
    )

    print(f"  Bbox filter: {inside.sum()} / {len(y_coords)} pixels inside")
    return y_coords[inside], x_coords[inside]


def filter_by_rate(y_coords, x_coords, cube, t_days, mode, threshold_mm_yr):
    """Keeps pixels whose linear displacement rate passes a threshold.

    Uses vectorized least-squares regression across all pixels simultaneously.
    Mode 'stable' keeps |rate| <= threshold; 'subsidence' keeps rate <= -threshold.

    Returns:
        tuple: Filtered (y_coords, x_coords).
    """
    ts_matrix = cube[:, y_coords, x_coords]
    valid = ~np.isnan(ts_matrix)

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
    has_data = valid_counts >= 10

    if mode == "stable":
        keep_mask = has_data & (np.abs(rate_mm_yr) <= threshold_mm_yr)
    else:
        keep_mask = has_data & (rate_mm_yr <= -threshold_mm_yr)

    kept_y = y_coords[keep_mask]
    kept_x = x_coords[keep_mask]

    print(
        f"  Rate filter ({mode}, {threshold_mm_yr} mm/yr): {len(kept_y)} / {len(y_coords)}"
    )
    return kept_y, kept_x


def build_displacement_cube(datasets, dates, ref_dates, row_slice=None, col_slice=None):
    """Builds a stitched 3D displacement cube from short_wavelength_displacement.

    OPERA changes the reference epoch roughly every 15 acquisitions. This function
    identifies contiguous segments sharing the same reference, computes per-pixel
    offsets at each boundary, and applies cumulative offsets to produce a continuous
    time series relative to the first reference epoch.

    Stitching strategy:
      - If consecutive segments share an acquisition date: exact per-pixel offset.
      - If no overlap: boundary stitch using last/first dates (~12-day gap error).
      - Pixels with NaN at the boundary get the spatial median offset as fallback.

    Returns:
        numpy.ndarray: Stitched cube of shape (T, H, W) in float64.
    """
    if row_slice and col_slice:
        raw = np.array(
            [
                ds["short_wavelength_displacement"].values.squeeze()[
                    row_slice, col_slice
                ]
                for ds in datasets
            ],
            dtype=np.float32,
        )
    else:
        raw = np.array(
            [ds["short_wavelength_displacement"].values.squeeze() for ds in datasets],
            dtype=np.float32,
        )

    # Identify segment boundaries
    segments = []
    seg_start = 0
    for i in range(1, len(ref_dates)):
        if ref_dates[i] != ref_dates[seg_start]:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, len(ref_dates)))

    if len(segments) == 1:
        print(f"  Single reference epoch — no stitching needed.")
        return raw.astype(np.float64)

    print(f"  Stitching {len(segments)} reference segments...")
    stitched = raw.astype(np.float64)

    for s in range(1, len(segments)):
        prev_start, prev_end = segments[s - 1]
        curr_start, curr_end = segments[s]

        # Look for an overlapping acquisition date
        prev_date_map = {dates[i]: i for i in range(prev_start, prev_end)}
        overlap = None
        for i in range(curr_start, curr_end):
            if dates[i] in prev_date_map:
                overlap = (prev_date_map[dates[i]], i)
                break

        if overlap:
            prev_i, curr_i = overlap
            prev_vals = stitched[prev_i]
            curr_vals = raw[curr_i]
            print(f"    Segment {s}: exact overlap at {dates[curr_i]:%Y-%m-%d}")
        else:
            prev_vals = stitched[prev_end - 1]
            curr_vals = raw[curr_start]
            gap_days = (dates[curr_start] - dates[prev_end - 1]).days
            print(f"    Segment {s}: no overlap, boundary stitch ({gap_days}-day gap)")

        # Per-pixel offset
        both_valid = ~np.isnan(prev_vals) & ~np.isnan(curr_vals)
        offset = np.full(raw.shape[1:], np.nan, dtype=np.float64)
        offset[both_valid] = prev_vals[both_valid] - curr_vals[both_valid]

        valid_offsets = offset[both_valid]
        if len(valid_offsets) > 0:
            fill_val = np.nanmedian(valid_offsets)
            offset[np.isnan(offset)] = fill_val
            print(
                f"    Offset: median={fill_val * 1000:.2f}mm, "
                f"std={np.nanstd(valid_offsets) * 1000:.2f}mm, "
                f"valid={both_valid.sum():,}/{both_valid.size:,}"
            )
        else:
            offset[:] = 0.0
            print(f"    WARNING: no valid pixels for offset, using 0")

        for t in range(curr_start, curr_end):
            stitched[t] += offset

    return stitched

def extract_time_series(cube, y_coords, x_coords, dates, max_nan_frac=0.2):
    """Extracts per-pixel displacement time series, interpolating small gaps.

    Pixels with more than max_nan_frac missing values are skipped entirely.
    Remaining gaps are filled with linear interpolation between valid dates.

    Returns:
        list: Dicts with keys 'y', 'x', 'time_series', 't_days'.
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
    """Computes temporal features from a single displacement time series.

    Features (all in mm or mm/yr):
      - linear_rate_mm_yr: slope of linear fit
      - r_squared_linear: goodness of linear fit
      - total_disp_mm: end-to-end displacement
      - max_abs_disp_mm: peak absolute displacement
      - acceleration: quadratic coefficient (m/day^2)
      - r2_ratio_quad_lin: ratio of quadratic to linear R^2 (capped at 10)
      - seasonal_amp_mm: amplitude of annual sinusoidal component
      - residual_std_mm: scatter around the linear fit
      - autocorr_lag1: lag-1 autocorrelation
      - vel_sign_changes: direction reversals in velocity
      - vel_kurtosis, vel_skewness: velocity distribution shape

    Returns:
        dict or None if fewer than 10 data points.
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
    f["r2_ratio_quad_lin"] = min(r2_quad / (lr.rvalue**2 + 1e-10), 10.0)

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
    """Computes a spatially strided rate map via vectorized linear regression.

    The stride reduces memory by stride^2 (e.g., 25x for stride=5). The rate
    map is used for neighborhood spatial features and gradient computation.

    Returns:
        numpy.ndarray: 2D rate map in m/day at reduced resolution.
    """
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
    """Computes spatial context features from the strided rate map neighborhood.

    Features:
      - mean_coherence: temporal coherence at this pixel
      - nbr_mean_rate_mm_yr: mean displacement rate in ~500m neighborhood
      - nbr_std_rate_mm_yr: rate variability in neighborhood
      - spatial_gradient: magnitude of rate gradient (m/day per meter)

    Returns:
        dict of spatial features.
    """
    H, W = shape
    f = {"mean_coherence": float(coherence[y, x])}

    y_s, x_s = y // stride, x // stride
    H_s, W_s = rate_map.shape
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

    if 1 <= y_s < H_s - 1 and 1 <= x_s < W_s - 1:
        spacing = 2 * stride * 30.0
        gy = (rate_map[y_s + 1, x_s] - rate_map[y_s - 1, x_s]) / spacing
        gx = (rate_map[y_s, x_s + 1] - rate_map[y_s, x_s - 1]) / spacing
        f["spatial_gradient"] = (
            float(np.sqrt(gy**2 + gx**2)) if not (np.isnan(gy) or np.isnan(gx)) else 0.0
        )
    else:
        f["spatial_gradient"] = 0.0

    return f


def load_dem_for_frame(dem_path, ds):
    """Reprojects a GLO-30 DEM GeoTIFF onto the exact OPERA pixel grid.

    Reads the CRS and pixel coordinates from the OPERA dataset, builds a
    matching rasterio transform, and bilinearly resamples the DEM to align
    pixel-for-pixel with the displacement data.

    Returns:
        numpy.ndarray: Aligned elevation grid (full frame, float32).
    """
    opera_crs = get_opera_crs(ds)
    x = ds.x.values
    y = ds.y.values

    res_x = x[1] - x[0]
    res_y = y[1] - y[0]

    transform = from_origin(
        x[0] - res_x / 2,
        y[0] - res_y / 2,
        abs(res_x),
        abs(res_y),
    )

    dst_shape = (len(y), len(x))
    aligned_dem = np.empty(dst_shape, dtype=np.float32)

    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned_dem,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=opera_crs,
            resampling=Resampling.bilinear,
        )

    print(
        f"  DEM reprojected: {aligned_dem.shape}, "
        f"range {np.nanmin(aligned_dem):.0f}–{np.nanmax(aligned_dem):.0f}m"
    )
    return aligned_dem


def compute_terrain(elevation, pixel_size=30.0):
    """Computes slope and aspect from a pixel-aligned DEM.

    Aspect is encoded as sin/cos pair to handle the circular wrap-around
    (359° and 1° are both north but numerically far apart as raw degrees).

    Returns:
        tuple: (slope_degrees, aspect_sin, aspect_cos)
    """
    dz_dy, dz_dx = np.gradient(elevation, pixel_size, pixel_size)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy)) % 360
    aspect_sin = np.sin(np.radians(aspect))
    aspect_cos = np.cos(np.radians(aspect))

    print(
        f"  Terrain: slope {np.nanmean(slope):.1f}° mean, "
        f"{np.nanmax(slope):.1f}° max"
    )
    return slope, aspect_sin, aspect_cos


def build_features(
    valid_pixels,
    coherence,
    rate_map,
    shape,
    label,
    region_name,
    frame_id,
    stride=5,
    slope=None,
    aspect_sin=None,
    aspect_cos=None,
):
    """Assembles temporal, spatial, and terrain features into a DataFrame.

    Iterates over valid pixels, computes all feature groups, and returns a
    table ready for classification. Each row is one pixel with its label.

    Returns:
        pandas.DataFrame with columns: metadata + temporal + spatial + terrain.
    """
    print(f"  Computing features for {len(valid_pixels)} pixels...")
    rows = []
    for i, px in enumerate(valid_pixels):
        tf = compute_temporal_features(px["time_series"], px["t_days"])
        if tf is None:
            continue
        sf = compute_spatial_features(
            px["y"], px["x"], coherence, rate_map, shape, stride=stride
        )
        row = {
            "pixel_y": px["y"],
            "pixel_x": px["x"],
            "frame_id": frame_id,
            "region": region_name,
            "label": label,
        }
        row.update(tf)
        row.update(sf)

        if slope is not None:
            row["slope_deg"] = float(slope[px["y"], px["x"]])
            row["aspect_sin"] = float(aspect_sin[px["y"], px["x"]])
            row["aspect_cos"] = float(aspect_cos[px["y"], px["x"]])

        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(valid_pixels)}")

    df = pd.DataFrame(rows)
    print(f"  Result: {len(df)} rows x {len(df.columns)} columns")
    return df


def save_outputs(df, valid_pixels, label, region_name):
    """Saves feature CSV and raw time series NPZ for a region.

    The CSV contains the feature table for training. The NPZ contains the
    raw time series arrays for models (e.g., RNN) that operate on sequences
    rather than extracted features.
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
    """Full preprocessing pipeline for one frame within a region.

    Steps:
      1. Load all granules and read reference epoch metadata
      2. Crop to region bbox (reduces memory from ~30 GB to <1 GB)
      3. Build displacement cube with cross-epoch stitching
      4. Load and crop DEM, compute terrain features
      5. Apply spatial filter (shapefile or bbox) then rate filter
      6. Extract time series, compute features, return DataFrame

    Returns:
        tuple: (DataFrame, list of pixel dicts) or (None, []) if no data survives.
    """
    print(f"\n  --- Frame {frame_id} ({len(file_list)} files) ---")
    dates, datasets, ref_dates = load_frame_datasets(file_list)

    try:
        slices = get_bbox_slices(datasets[0], info["bbox"])
        if slices is None:
            print(f"  Bbox doesn't intersect this frame.")
            return None, []
        row_slice, col_slice = slices

        x_utm = datasets[0].x.values[col_slice]
        y_utm = datasets[0].y.values[row_slice]

        good_mask, coherence = get_quality_mask(datasets[-1], row_slice, col_slice)

        label = info["label"]
        spacing = 90 if label == "landslide" else 500
        y_coords, x_coords = sample_pixels(good_mask, min_spacing_m=spacing)

        cube = build_displacement_cube(datasets, dates, ref_dates, row_slice, col_slice)

        dem_path = PROJECT_ROOT / "data" / "dem" / f"dem_{frame_id}.tif"
        if dem_path.exists():
            elevation = load_dem_for_frame(dem_path, datasets[0])
            elevation = elevation[row_slice, col_slice]
            slope, aspect_sin, aspect_cos = compute_terrain(elevation)
        else:
            print(f"  WARNING: No DEM at {dem_path}, terrain features will be 0")
            slope = np.zeros(cube.shape[1:], dtype=np.float32)
            aspect_sin = np.zeros_like(slope)
            aspect_cos = np.zeros_like(slope)

        t_days = np.array([(d - dates[0]).days for d in dates], dtype=float)

        # Step 1: Spatial filter
        if info.get("shapefile"):
            y_coords, x_coords = filter_by_shapefile(
                y_coords,
                x_coords,
                datasets[0],
                info,
                x_utm=x_utm,
                y_utm=y_utm,
            )
        else:
            y_coords, x_coords = filter_by_bbox(
                y_coords,
                x_coords,
                datasets[0],
                info["bbox"],
                x_utm=x_utm,
                y_utm=y_utm,
            )

        # Step 2: Rate filter
        if label == "stable":
            y_coords, x_coords = filter_by_rate(
                y_coords, x_coords, cube, t_days, "stable", 2.0
            )
        elif label == "subsidence":
            y_coords, x_coords = filter_by_rate(
                y_coords, x_coords, cube, t_days, "subsidence", 10.0
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
            slope=slope,
            aspect_sin=aspect_sin,
            aspect_cos=aspect_cos,
        )
        return df, valid_pixels

    finally:
        for ds in datasets:
            ds.close()


def process_region(region_name, max_files=None, max_samples=1000):
    """Processes all frames in a region and saves combined outputs.

    Returns:
        pandas.DataFrame or None.
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
    """Merges all regional feature CSVs into a single training dataset."""
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

    return combined



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPERA DISP-S1 Preprocessing")
    parser.add_argument("--region", type=str, help="Region to process")
    parser.add_argument("--combine", action="store_true", help="Combine all regions")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    if args.combine:
        combine_all()
    elif args.region:
        process_region(args.region, args.max_files, args.max_samples)
    else:
        print(
            "Usage:\n  python preprocess.py --region san_andreas\n  python preprocess.py --combine"
        )

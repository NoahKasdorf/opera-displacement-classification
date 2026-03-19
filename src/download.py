"""Download OPERA DISP-S1 granules via earthaccess."""

import earthaccess
from dotenv import load_dotenv
import argparse

from config import PROJECT_ROOT, REGIONS

TARGET_FRAMES = {
    "washington": "16950",
    "hayward": "09156",
    "oregon_coast": "03323",
    "san_andreas": "09155",
    "san_joaquin": "11116",
    "houston": "38238",
    "california_north": "09158",
    "wasatch": "05131",
    "phoenix": "05126",
    "st_helens": "03322",
    
    "stable_kansas": "08889",
    "stable_texas": "28482",
    "stable_sierra": "16942",
}


def download_region(region_name, max_files=100):
    """Downloads up to max_files OPERA DISP-S1 granules for a given region.

    Args:
        region_name (str): The name of the region defined in config.py.
        max_files (int, optional): Maximum number of granules to fetch. Defaults to 100.
    """
    info = REGIONS.get(region_name)
    frame_id = TARGET_FRAMES.get(region_name)
    if not info:
        print(f"Error: Region '{region_name}' not found.")
        return

    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        print("Error: Failed to authenticate with Earthdata.")
        return

    out_dir = PROJECT_ROOT / "data" / region_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Targeting {region_name} | Frame: {frame_id} | Goal: {max_files} files")

    bbox = info["bbox"]
    print(f"Searching OPERA DISP-S1 for {region_name} | bbox={bbox}")

    results = earthaccess.search_data(
        short_name="OPERA_L3_DISP-S1_V1",
        bounding_box=bbox,
        granule_name= f"*_F{frame_id}_*",
        temporal=("2016-07-01", "2026-03-17"),
        #temporal=("2022-01-01", "2026-06-01"),
        count=max_files,
        sort_key="-start_date",
    )
    print(f"Found {len(results)} granules.")

    if results:
        # Extract ONLY the .nc file links to bypass broken sidecar files
        nc_urls = []
        for granule in results:
            nc_urls.extend(
                [link for link in granule.data_links() if link.endswith(".nc")]
            )

        print(f"Downloading {len(nc_urls)} NetCDF files to {out_dir}...")
        earthaccess.download(nc_urls, local_path=str(out_dir))
    else:
        print("No files found.")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download OPERA DISP-S1 data")
    parser.add_argument("--region", type=str, help="Single region to download")
    parser.add_argument("--all", action="store_true", help="Download all regions")
    parser.add_argument("--max-files", type=int, default=100)
    args = parser.parse_args()

    if args.all:
        for name in REGIONS:
            print(f"\n{'='*40}\nRegion: {name}\n{'='*40}")
            download_region(name, args.max_files)
    elif args.region:
        download_region(args.region, args.max_files)
    else:
        print(
            "Usage:\n  python download.py --region stable_texas\n  python download.py --all"
        )

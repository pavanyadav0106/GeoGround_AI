"""
download_nwdp.py
================
GeoGround AI — Phase 0: Download NWDP Groundwater Data

Attempts to fetch the CGWB "Ground Water Level (Manual – Quarterly)"
dataset for Telangana from the National Water Data Portal (NWDP).

The NWDP exposes a CKAN-compatible Data API. This script tries:
  1. NWDP dataset page for Telangana GW Level
  2. India-WRIS Swagger API (if available)
  3. Falls back with clear instructions if downloads fail

Usage:
    python scripts/download_nwdp.py
    python scripts/download_nwdp.py --state Telangana
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# ─── NWDP Dataset URLs ──────────────────────────────────────────────────────
# These are the known dataset resource IDs and URLs on nwdp.nwic.gov.in
# The NWDP uses CKAN under the hood; datasets are accessible via:
# https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id=<ID>

NWDP_BASE        = "https://nwdp.nwic.gov.in"
NWDP_SEARCH_API  = f"{NWDP_BASE}/api/3/action/package_search"
NWDP_DATASTORE   = f"{NWDP_BASE}/api/3/action/datastore_search"

# Known dataset names on NWDP for GW Level
NWDP_GW_SEARCH_TERMS = [
    "Ground Water Level Telangana",
    "Groundwater Level CGWB Telangana",
    "GWL Manual Quarterly Telangana",
    "groundwater level telangana cgwb",
]

# ─── VERIFIED Direct Download URLs ──────────────────────────────────────────
# Verified on 2026-08-16 by browser inspection of nwdp.nwic.gov.in
# Dataset: Ground Water Level (Manual - Quarterly), Telangana GW Department
# Dataset ID: 10cce234-b474-4870-aa37-42169296442b
# Login required: No — publicly accessible without authentication

NWDP_VERIFIED_RESOURCES = [
    {
        "name": "gwl_manual_quarterly_telangana_1991_2020",
        "resource_id": "b6df164e-e86c-43a1-837f-bd94f846b694",
        "url": (
            "https://nwdp.nwic.gov.in/dataset/"
            "10cce234-b474-4870-aa37-42169296442b/resource/"
            "b6df164e-e86c-43a1-837f-bd94f846b694/download/"
            "gwl_manual_quarterly_telangana-gw_ts_1991_2020.csv"
        ),
        "period": "1991-2020",
    },
    {
        "name": "gwl_manual_quarterly_telangana_2021_2025",
        "resource_id": "1290f7be-81f7-4d21-88f6-c608c8800b57",
        "url": (
            "https://nwdp.nwic.gov.in/dataset/"
            "10cce234-b474-4870-aa37-42169296442b/resource/"
            "1290f7be-81f7-4d21-88f6-c608c8800b57/download/"
            "gwl_manual_quarterly_telangana_gw_ts_2021_2025.csv"
        ),
        "period": "2021-2025",
    },
    {
        "name": "gwl_manual_quarterly_telangana_2026_2030",
        "resource_id": "11dfacb3-a801-4257-aa05-b0d79c669764",
        "url": (
            "https://nwdp.nwic.gov.in/dataset/"
            "10cce234-b474-4870-aa37-42169296442b/resource/"
            "11dfacb3-a801-4257-aa05-b0d79c669764/download/"
            "gwl_manual_quarterly_telangana_gw_ts_2026_2030.csv"
        ),
        "period": "2026-2030 (likely sparse)",
    },
]

import sys
import io
# Force UTF-8 output on Windows to avoid cp1252 emoji encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HEADERS = {
    "User-Agent": "GeoGround-AI-Research/1.0 (Academic Project)",
    "Accept": "application/json, text/csv",
}

TIMEOUT = 30  # seconds


def search_nwdp_datasets(term: str) -> list:
    """Search NWDP for datasets matching `term`."""
    try:
        resp = requests.get(
            NWDP_SEARCH_API,
            params={"q": term, "rows": 10},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data["result"]["results"]
    except requests.RequestException as e:
        print(f"  NWDP search failed for '{term}': {e}")
    return []


def try_download_resource(resource: dict, out_dir: Path) -> bool:
    """Attempt to download a NWDP resource file (CSV preferred)."""
    url  = resource.get("url", "")
    fmt  = resource.get("format", "").upper()
    name = resource.get("name", "resource")

    if not url:
        return False

    # Prefer CSV
    if fmt not in ("CSV", ""):
        print(f"    Skipping non-CSV resource: {fmt} — {name}")
        return False

    filename = f"nwdp_{name.lower().replace(' ', '_')[:60]}.csv"
    out_path = out_dir / filename

    if out_path.exists():
        print(f"  ✅ Already downloaded: {out_path.name}")
        return True

    print(f"  Downloading: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        resp.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = out_path.stat().st_size / 1024
        print(f"  ✅ Saved: {out_path.name}  ({size_kb:.1f} KB)")
        return True

    except requests.RequestException as e:
        print(f"  ❌ Download failed: {e}")
        if out_path.exists():
            out_path.unlink()
    return False


def try_nwdp_datastore(resource_id: str, out_dir: Path, state: str = "Telangana") -> bool:
    """
    Try fetching data from NWDP CKAN Datastore API by resource_id.
    Returns True if any data was saved.
    """
    offset = 0
    limit  = 10000
    all_records = []

    print(f"  Fetching datastore resource {resource_id} (paginated)…")

    while True:
        try:
            resp = requests.get(
                NWDP_DATASTORE,
                params={
                    "resource_id": resource_id,
                    "limit": limit,
                    "offset": offset,
                },
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("success"):
                break

            records = data["result"]["records"]
            if not records:
                break

            all_records.extend(records)
            total = data["result"]["total"]
            offset += limit
            print(f"  Fetched {len(all_records):,} / {total:,} records", end="\r")

            if offset >= total:
                break

            time.sleep(0.5)  # be polite to the server

        except requests.RequestException as e:
            print(f"\n  Datastore fetch error at offset {offset}: {e}")
            break

    if all_records:
        import pandas as pd
        df = pd.DataFrame(all_records)
        out_path = out_dir / f"nwdp_gwl_{state.lower()}_datastore.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  ✅ Saved {len(df):,} records: {out_path.name}")
        return True

    print("\n  Datastore returned no records.")
    return False


def print_manual_instructions():
    """Print fallback instructions if all automated downloads fail."""
    print("\n" + "─" * 65)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("─" * 65)
    print("""
  The automated NWDP download did not succeed.
  Please download the data manually:

  ┌─────────────────────────────────────────────────────────┐
  │  Option 1 — National Water Data Portal (NWDP)          │
  │  URL: https://nwdp.nwic.gov.in/                         │
  │  Steps:                                                  │
  │   1. Go to the portal and search for:                   │
  │      "Ground Water Level Manual Quarterly Telangana"    │
  │   2. Click the matching dataset                          │
  │   3. Under "Data and Resources", click Download (CSV)   │
  │   4. Save the file to: ml/data/raw/                     │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  Option 2 — India-WRIS                                  │
  │  URL: https://indiawris.gov.in/                         │
  │  Steps:                                                  │
  │   1. Log in (free registration)                          │
  │   2. Navigate to Groundwater → Data                     │
  │   3. Filter: State = Telangana, District = Hyderabad    │
  │   4. Download all available years as CSV                 │
  │   5. Save to: ml/data/raw/                              │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  Option 3 — CGWB Ground Water Year Book                 │
  │  URL: https://cgwb.gov.in/                              │
  │  Steps:                                                  │
  │   1. Publications → Ground Water Year Book → Telangana  │
  │   2. Download the PDF/data annex                         │
  │   3. Extract the monitoring well table manually          │
  └─────────────────────────────────────────────────────────┘

  Once the CSV is saved, re-run:
    python src/data_loader.py --validate --district Hyderabad
""")


def main():
    parser = argparse.ArgumentParser(description="Download NWDP groundwater data")
    parser.add_argument("--state",   default="Telangana", help="State name")
    parser.add_argument("--outdir",  default=str(RAW_DATA_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  GeoGround AI — NWDP Data Downloader")
    print(f"  State: {args.state}")
    print(f"  Output: {out_dir}\n")

    success = False

    # ── Strategy 0: Use verified direct URLs (fastest, most reliable) ─
    print("  Strategy 0: Trying verified direct download URLs…")
    for resource in NWDP_VERIFIED_RESOURCES:
        print(f"  Period: {resource['period']}")
        if try_download_resource(
            {"url": resource["url"], "format": "CSV", "name": resource["name"]},
            out_dir
        ):
            success = True  # continue downloading all periods

    if success:
        print(f"\n  ✅ Direct downloads complete.")
        return

    # ── Strategy 1: Search and download from NWDP ─────────────
    print("  Strategy 1: Searching NWDP datasets…")
    for term in NWDP_GW_SEARCH_TERMS:
        print(f"  Searching: '{term}'")
        datasets = search_nwdp_datasets(term)

        for ds in datasets:
            print(f"  Found dataset: {ds.get('title', 'N/A')}")
            for resource in ds.get("resources", []):
                if try_download_resource(resource, out_dir):
                    success = True
                    break
                # Also try datastore if resource_id exists
                rid = resource.get("id")
                if rid:
                    if try_nwdp_datastore(rid, out_dir, state=args.state):
                        success = True
                        break

        if success:
            break
        time.sleep(1)

    # ── Strategy 2: Known resource IDs (may change over time) ─
    if not success:
        print("\n  Strategy 2: Trying NWDP Datastore API with verified resource IDs…")
        KNOWN_RESOURCE_IDS = [
            r["resource_id"] for r in NWDP_VERIFIED_RESOURCES
        ]
        for rid in KNOWN_RESOURCE_IDS:
            if try_nwdp_datastore(rid, out_dir, state=args.state):
                success = True
                break

    if not success:
        print("\n  ⚠️  Automated download was not successful.")
        print_manual_instructions()
        sys.exit(1)
    else:
        print(f"\n  ✅ Download complete. Files saved to: {out_dir}")
        print(f"  Next step: python src/data_loader.py --validate --district Hyderabad")


if __name__ == "__main__":
    main()

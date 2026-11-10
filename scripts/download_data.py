"""Download the raw data for build.py. No API key needed.

In:   Puerto Rico Institute of Statistics (IEPR): monthly exports by 10-digit product and
      destination since July 2002, built by IEPR from U.S. Census Bureau records. Unlike the
      Census API it also contains shipments to the U.S. (mainland and U.S. Virgin Islands).
      Atlas of Economic Complexity (Harvard Growth Lab): world exports and PCI per product.
Out:  data/raw/iepr/pr_exports_hs4.csv      year x 4-digit product x destination (foreign / us)
      data/raw/iepr/HTS10_ALL.zip           the original download (cached, not in git)
      data/raw/atlas/products_hs92.csv, data/raw/atlas/product_year_hs92.csv

Run from the repository folder:  python scripts/download_data.py   (about 5 minutes)
"""
import io
import struct
import time
import zipfile
import zlib
from pathlib import Path

import pandas as pd
import requests

HEADERS = {"User-Agent": "RUG Data Wise student project (Puerto Rico export complexity)"}

# "Base de datos detallados de comercio externo" on
# https://apps.estadisticas.pr/iepr/Publicaciones/Proyectosespeciales/ComercioExterno.aspx
# If this link stops working, copy the new link of that item from the page.
IEPR_ZIP_URL = ("http://apps.estadisticas.pr/iepr/LinkClick.aspx?"
                "fileticket=JVyYmIHqbqc%3d&tabid=284&mid=244930")
US_DESTINATIONS = ["United States", "Virgin Islands of the United States"]  # all other countries = foreign

ATLAS_API = "https://atlas.hks.harvard.edu/api/graphql"
ATLAS_YEARS = range(1995, 2025)  # Atlas PCI is available 1995-2024


# ---------------------------------------------------------------- IEPR (Puerto Rico)

def download(url: str, path: Path) -> Path:
    """Download once; later runs reuse the file. Delete it to force a fresh download."""
    if path.exists():
        print(f"  using cached {path.name}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} ...")
    tmp = path.with_suffix(".part")
    with requests.get(url, headers=HEADERS, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(path)
    return path


def read_zip_entry(blob: bytes, name: str) -> bytes:
    """One file from a zip. The IEPR zip lacks its table of contents at the end, so normal
    zip readers fail; then we walk the entries from the start of the file instead."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            return z.read(next(n for n in z.namelist() if n.endswith(name)))
    except (zipfile.BadZipFile, StopIteration):
        pass
    pos = 0
    while blob[pos:pos + 4] == b"PK\x03\x04":  # start of an entry
        method = struct.unpack("<H", blob[pos + 8:pos + 10])[0]
        size = struct.unpack("<I", blob[pos + 18:pos + 22])[0]  # 0 = unknown, stored after the data
        name_len, extra_len = struct.unpack("<HH", blob[pos + 26:pos + 30])
        entry = blob[pos + 30:pos + 30 + name_len].decode()
        start = pos + 30 + name_len + extra_len
        if method == 8:  # compressed: decompress until the stream ends
            d = zlib.decompressobj(-15)
            data = d.decompress(blob[start:start + size] if size else blob[start:])
            end = start + size if size else len(blob) - len(d.unused_data)
        else:
            data, end = blob[start:start + size], start + size
        if entry.endswith(name):
            return data
        pos = blob.find(b"PK\x03\x04", end)
        if pos < 0:
            break
    raise RuntimeError(f"{name} not found in the IEPR zip")


def iepr_exports(zip_path: Path) -> pd.DataFrame:
    """Monthly 10-digit product x country data -> year x HS4 x destination group (US$)."""
    inner = read_zip_entry(zip_path.read_bytes(), "EXPORT_HTS10_ALL.zip")
    df = pd.read_csv(io.BytesIO(read_zip_entry(inner, "EXPORT_HTS10_ALL.csv")),
                     usecols=["country", "year", "month", "value", "HTS"], dtype={"country": str, "HTS": str})
    df["hs4"] = df.HTS.str.lstrip("'").str.zfill(10).str[:4]  # codes are stored as text like '0101100000
    df["destination"] = df.country.isin(US_DESTINATIONS).map({True: "us", False: "foreign"})
    months = df.groupby("year").month.nunique().rename("months_reported")  # lets build.py drop partial years
    out = (df.groupby(["year", "hs4", "destination"], as_index=False)["value"].sum()
             .rename(columns={"value": "value_usd"}).join(months, on="year"))
    for (y, d), v in out.groupby(["year", "destination"]).value_usd.sum().items():
        print(f"  {y} {d:8s} ${v / 1e9:6.2f}B")
    return out


# ---------------------------------------------------------------- Atlas (world)

def gql(query: str) -> list:
    """Run one GraphQL query and return the first field of `data`. Fails loud on errors."""
    r = requests.post(ATLAS_API, json={"query": query}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(f"Atlas API error: {body['errors']}")
    time.sleep(0.6)  # stay below the API's limit of 120 requests per minute
    return next(iter(body["data"].values()))


def atlas_products() -> pd.DataFrame:
    """HS92 4-digit products: code, name, sector, good or service."""
    rows = gql("{ productHs92(productLevel:4) { productId code nameShortEn productType topParent { nameShortEn } } }")
    return pd.json_normalize(rows).rename(columns={"topParent.nameShortEn": "sector"})


def atlas_product_year() -> pd.DataFrame:
    """World export value and PCI per product per year."""
    frames = []
    for y in ATLAS_YEARS:
        rows = gql(f"{{ productYear(productClass:HS92, productLevel:4, yearMin:{y}, yearMax:{y}) "
                   "{ productId year exportValue pci } }")
        frames.append(pd.DataFrame(rows))
        print(f"  Atlas {y}: {len(rows)} products")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- main

def main() -> None:
    root = Path(__file__).resolve().parents[1]  # the repository folder
    iepr_dir, atlas_dir = root / "data" / "raw" / "iepr", root / "data" / "raw" / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    print("Puerto Rico exports (IEPR) ...")
    zip_path = download(IEPR_ZIP_URL, iepr_dir / "HTS10_ALL.zip")
    iepr_exports(zip_path).to_csv(iepr_dir / "pr_exports_hs4.csv", index=False)
    print("World trade and PCI (Atlas) ...")
    atlas_products().to_csv(atlas_dir / "products_hs92.csv", index=False)
    atlas_product_year().to_csv(atlas_dir / "product_year_hs92.csv", index=False)
    print(f"Done -> {root / 'data' / 'raw'}")


if __name__ == "__main__":
    main()

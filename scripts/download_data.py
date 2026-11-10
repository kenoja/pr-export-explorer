import io
import struct
import time
import zipfile
import zlib
from pathlib import Path

import pandas as pd
import requests

# identify ourselves to the data servers
HEADERS = {"User-Agent": "RUG Data Wise student project (Puerto Rico export complexity)"}


# Puerto Rico exports (IEPR)

# download once, later runs reuse the file (delete it to download again)
def download(url: str, path: Path) -> Path:
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


# read one file from a zip
# (the IEPR zip misses its table of contents, so if the normal reader fails we walk the entries)
def read_zip_entry(blob: bytes, name: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            return z.read(next(n for n in z.namelist() if n.endswith(name)))
    except (zipfile.BadZipFile, StopIteration):
        pass
    pos = 0
    while blob[pos:pos + 4] == b"PK\x03\x04":  # start of an entry
        method = struct.unpack("<H", blob[pos + 8:pos + 10])[0]
        size = struct.unpack("<I", blob[pos + 18:pos + 22])[0]  # (0 = unknown)
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


# monthly product x country data -> year x HS4 x destination (foreign / us)
def iepr_exports(zip_path: Path) -> pd.DataFrame:
    inner = read_zip_entry(zip_path.read_bytes(), "EXPORT_HTS10_ALL.zip")
    df = pd.read_csv(io.BytesIO(read_zip_entry(inner, "EXPORT_HTS10_ALL.csv")),
                     usecols=["country", "year", "month", "value", "HTS"], dtype={"country": str, "HTS": str})
    df["hs4"] = df.HTS.str.lstrip("'").str.zfill(10).str[:4]  # (codes stored as text like '0101100000)
    us = df.country.isin(["United States", "Virgin Islands of the United States"])
    df["destination"] = us.map({True: "us", False: "foreign"})
    months = df.groupby("year").month.nunique().rename("months_reported")  # (build.py drops partial years)
    out = (df.groupby(["year", "hs4", "destination"], as_index=False)["value"].sum()
             .rename(columns={"value": "value_usd"}).join(months, on="year"))
    for (y, d), v in out.groupby(["year", "destination"]).value_usd.sum().items():
        print(f"  {y} {d:8s} ${v / 1e9:6.2f}B")
    return out


# world trade and PCI (Atlas)

# one GraphQL query, stops on errors
def gql(query: str) -> list:
    r = requests.post("https://atlas.hks.harvard.edu/api/graphql", json={"query": query},
                      headers=HEADERS, timeout=120)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(f"Atlas API error: {body['errors']}")
    time.sleep(0.6)  # (API limit: 120 requests per minute)
    return next(iter(body["data"].values()))


# product list: code, name, sector, good or service
def atlas_products() -> pd.DataFrame:
    rows = gql("{ productHs92(productLevel:4) { productId code nameShortEn productType topParent { nameShortEn } } }")
    return pd.json_normalize(rows).rename(columns={"topParent.nameShortEn": "sector"})


# world export value and PCI per product per year (PCI available 1995-2024)
def atlas_product_year() -> pd.DataFrame:
    frames = []
    for y in range(1995, 2025):
        rows = gql(f"{{ productYear(productClass:HS92, productLevel:4, yearMin:{y}, yearMax:{y}) "
                   "{ productId year exportValue pci } }")
        frames.append(pd.DataFrame(rows))
        print(f"  Atlas {y}: {len(rows)} products")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    # paths (repository folder = one level above this script)
    root = Path(__file__).resolve().parents[1]
    iepr_dir = root / "data" / "raw" / "iepr"
    atlas_dir = root / "data" / "raw" / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    # "Base de datos detallados de comercio externo" on
    # https://apps.estadisticas.pr/iepr/Publicaciones/Proyectosespeciales/ComercioExterno.aspx
    # (if the link stops working, copy the new one from that page)
    iepr_url = "http://apps.estadisticas.pr/iepr/LinkClick.aspx?fileticket=JVyYmIHqbqc%3d&tabid=284&mid=244930"

    print("Puerto Rico exports (IEPR) ...")
    zip_path = download(iepr_url, iepr_dir / "HTS10_ALL.zip")
    iepr_exports(zip_path).to_csv(iepr_dir / "pr_exports_hs4.csv", index=False)
    print("World trade and PCI (Atlas) ...")
    atlas_products().to_csv(atlas_dir / "products_hs92.csv", index=False)
    atlas_product_year().to_csv(atlas_dir / "product_year_hs92.csv", index=False)
    print(f"Done -> {root / 'data' / 'raw'}")


if __name__ == "__main__":
    main()

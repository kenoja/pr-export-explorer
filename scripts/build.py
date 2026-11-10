"""Build the Puerto Rico export dataframe and the interactive visualization.

In:   data/raw/iepr/pr_exports_hs4.csv      (from download_data.py)
      data/raw/atlas/products_hs92.csv, data/raw/atlas/product_year_hs92.csv
      scripts/template.html, fonts/*.woff2
Out:  data/pr_exports.csv                   one row per product per year
      pr_exports.html                       the two linked charts (open in a browser, works offline)

Run from the repository folder:  python scripts/build.py
"""
import base64
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

MIN_VALUE_M = 0.1  # drop product-years below $0.1M: mostly one-off shipments

# Sector colours from the V2A palette (v2aconsulting.com): the three biggest export sectors get
# the three brand colours (orange, bahia blue, celeste); the others get tints and shades of them.
SECTOR_COLORS = {
    "Chemicals": "#FF8C6B", "Machinery": "#1D4289", "Electronics": "#41B5E5",
    "Agriculture": "#BE6A50", "Minerals": "#0D2145", "Metals": "#768DB7",
    "Textiles": "#3188AC", "Stone": "#FFBAA5", "Vehicles": "#8CD2EF", "Other": "#6C757D",
}

# Inter (free, SIL Open Font License) stands in for V2A's body font, which is licensed and
# can't be shared. The files are embedded in the page so it also looks right offline.
FONTS = {400: "Inter-Regular.woff2", 600: "Inter-SemiBold.woff2"}


# ---------------------------------------------------------------- load

def load_pr_exports(raw: Path) -> pd.DataFrame:
    """Puerto Rico exports per year x HS4 product, one column per destination group (US$)."""
    df = pd.read_csv(raw / "iepr" / "pr_exports_hs4.csv", dtype={"hs4": str})
    complete = df[df.months_reported == 12]  # 2002 and the current year are partial years
    wide = (complete.pivot_table(index=["year", "hs4"], columns="destination",
                                 values="value_usd", aggfunc="sum", fill_value=0)
                    .reset_index().rename_axis(columns=None))
    wide["total"] = wide.foreign + wide.us
    return wide


def load_world(raw: Path) -> pd.DataFrame:
    """World exports and PCI per year x HS92 product (goods only), from the Atlas."""
    products = pd.read_csv(raw / "atlas" / "products_hs92.csv", dtype=str)
    product_year = pd.read_csv(raw / "atlas" / "product_year_hs92.csv")
    world = product_year.merge(products, on="productId")
    world = world[world.productType == "good"]
    return world.rename(columns={"exportValue": "world"})[["year", "code", "nameShortEn", "sector", "world", "pci"]]


# ---------------------------------------------------------------- transform

def rca(value: pd.Series, total_by_year: pd.Series, world: pd.Series, world_by_year: pd.Series,
        year: pd.Series) -> pd.Series:
    """Balassa RCA = (Puerto Rico's share of product p) / (world's share of product p)."""
    return (value / year.map(total_by_year)) / (world / year.map(world_by_year))


def build_dataframe(pr: pd.DataFrame, world: pd.DataFrame) -> pd.DataFrame:
    years = sorted(set(pr.year) & set(world.year))
    pr, world = pr[pr.year.isin(years)], world[world.year.isin(years)]

    # Puerto Rico totals include every product, also the few that don't match an Atlas code.
    pr_totals = pr.groupby("year")[["foreign", "total"]].sum()
    world_total = world.groupby("year").world.sum()

    df = pr.merge(world, left_on=["year", "hs4"], right_on=["year", "code"], how="left")
    matched = df[df.code.notna()].groupby("year").total.sum() / pr_totals.total
    print(f"Years {years[0]}-{years[-1]}; export value matched to Atlas codes: "
          f"{matched.min():.1%} (worst year) to {matched.max():.1%} (best year)")
    df = df[df.code.notna()].copy()

    df["rca_foreign"] = rca(df.foreign, pr_totals.foreign, df.world, world_total, df.year)
    df["rca_total"] = rca(df.total, pr_totals.total, df.world, world_total, df.year)
    out = pd.DataFrame({
        "year": df.year, "code": df.hs4, "productName": df.nameShortEn, "sector": df.sector,
        "exports_foreign_M": (df.foreign / 1e6).round(3), "exports_us_M": (df.us / 1e6).round(3),
        "exports_total_M": (df.total / 1e6).round(3), "worldTrade_M": (df.world / 1e6).round(1),
        "rca_foreign": df.rca_foreign.round(4), "rca_total": df.rca_total.round(4),
        "valuePCI": df.pci.round(4),
    })
    out = out[(out.exports_total_M >= MIN_VALUE_M) & out.valuePCI.notna()]
    return out.sort_values(["year", "code"]).reset_index(drop=True)


def check(df: pd.DataFrame) -> None:
    """Fail loud if the dataframe breaks basic expectations."""
    assert not df.duplicated(["year", "code"]).any(), "duplicate product-year rows"
    gap = (df.exports_total_M - df.exports_foreign_M - df.exports_us_M).abs()
    assert (gap <= 0.002).all(), "total != foreign + U.S. (beyond rounding)"
    unknown = set(df.sector) - set(SECTOR_COLORS)
    assert not unknown, f"sectors without a colour: {unknown}"
    for y, g in df.groupby("year"):
        print(f"  {y}: {len(g):4d} products, foreign ${g.exports_foreign_M.sum() / 1e3:5.1f}B, "
              f"U.S. ${g.exports_us_M.sum() / 1e3:5.1f}B")


# ---------------------------------------------------------------- visualization

def page_data(df: pd.DataFrame) -> dict:
    """Compact, column-wise data for the page (keeps the HTML file small)."""
    products = df.drop_duplicates("code").sort_values("code")[["code", "productName", "sector"]]
    index = {c: i for i, c in enumerate(products.code)}
    by_sector = df.groupby("sector").exports_total_M.sum().sort_values(ascending=False)
    return {
        "years": sorted(int(y) for y in df.year.unique()),
        "sectors": [{"name": s, "color": SECTOR_COLORS[s]} for s in by_sector.index],  # biggest first
        "products": products.values.tolist(),
        "rows": {
            "product": df.code.map(index).tolist(), "year": df.year.tolist(),
            "foreign": df.exports_foreign_M.tolist(), "us": df.exports_us_M.tolist(),
            "rcaForeign": df.rca_foreign.tolist(), "rcaTotal": df.rca_total.tolist(),
            "pci": df.valuePCI.tolist(),
        },
    }


def font_css(fonts_dir: Path) -> str:
    """@font-face rules with the font files embedded as data URLs."""
    rules = []
    for weight, file in FONTS.items():
        data = base64.b64encode((fonts_dir / file).read_bytes()).decode()
        rules.append(f"@font-face {{ font-family: 'Inter'; font-weight: {weight}; "
                     f"src: url(data:font/woff2;base64,{data}) format('woff2'); }}")
    return "\n".join(rules)


def write_html(df: pd.DataFrame, template: Path, fonts_dir: Path, out: Path) -> None:
    html = (template.read_text(encoding="utf-8")
            .replace("/*FONTS*/", font_css(fonts_dir))
            .replace("/*PLOTLY_JS*/", get_plotlyjs())
            .replace("/*DATA*/null", json.dumps(page_data(df), separators=(",", ":"))))
    out.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- main

def main() -> None:
    root = Path(__file__).resolve().parents[1]  # the repository folder
    raw = root / "data" / "raw"
    out_csv, out_html = root / "data" / "pr_exports.csv", root / "pr_exports.html"

    df = build_dataframe(load_pr_exports(raw), load_world(raw))
    check(df)
    df.to_csv(out_csv, index=False)
    write_html(df, root / "scripts" / "template.html", root / "fonts", out_html)
    print(f"Dataframe: {len(df)} rows -> {out_csv}\nVisualization -> {out_html}")


if __name__ == "__main__":
    main()

import base64
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

# set colours (V2A palette)
SECTOR_COLORS = {
    "Chemicals": "#FF8C6B",
    "Machinery": "#1D4289",
    "Electronics": "#41B5E5",
    "Agriculture": "#BE6A50",
    "Minerals": "#0D2145",
    "Metals": "#768DB7",
    "Textiles": "#3188AC",
    "Stone": "#FFBAA5",
    "Vehicles": "#8CD2EF",
    "Other": "#6C757D",
}


# load data

def load_pr_exports(raw_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_dir / "iepr" / "pr_exports_hs4.csv", dtype={"hs4": str})
    complete = raw[raw.months_reported == 12]  # (some partial years)
    wide = (complete.pivot_table(index=["year", "hs4"], columns="destination",
                                 values="value_usd", aggfunc="sum", fill_value=0)
                    .reset_index().rename_axis(columns=None))
    wide["total"] = wide.foreign + wide.us
    return wide


def load_world(raw_dir: Path) -> pd.DataFrame:
    products = pd.read_csv(raw_dir / "atlas" / "products_hs92.csv", dtype=str)
    product_year = pd.read_csv(raw_dir / "atlas" / "product_year_hs92.csv")
    world = product_year.merge(products, on="productId")
    world = world[world.productType == "good"]
    return world.rename(columns={"exportValue": "world"})[["year", "code", "nameShortEn", "sector", "world", "pci"]]


# transform

# RCA = PR share of product / world's share of product
def rca(value: pd.Series, total_by_year: pd.Series, world: pd.Series, world_by_year: pd.Series,
        year: pd.Series) -> pd.Series:
    return (value / year.map(total_by_year)) / (world / year.map(world_by_year))


def build_dataframe(pr: pd.DataFrame, world: pd.DataFrame) -> pd.DataFrame:
    years = sorted(set(pr.year) & set(world.year))
    pr, world = pr[pr.year.isin(years)], world[world.year.isin(years)]

    # Puerto Rico totals include every product
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
    out = out[(out.exports_total_M >= 0.1) & out.valuePCI.notna()]
    return out.sort_values(["year", "code"]).reset_index(drop=True)

# check dataframe
def check(df: pd.DataFrame) -> None:
    assert not df.duplicated(["year", "code"]).any(), "duplicate product-year rows"
    gap = (df.exports_total_M - df.exports_foreign_M - df.exports_us_M).abs()
    assert (gap <= 0.002).all(), "total != foreign + U.S. (beyond rounding)"
    unknown = set(df.sector) - set(SECTOR_COLORS)
    assert not unknown, f"sectors without a colour: {unknown}"
    for y, g in df.groupby("year"):
        print(f"  {y}: {len(g):4d} products, foreign ${g.exports_foreign_M.sum() / 1e3:5.1f}B, "
              f"U.S. ${g.exports_us_M.sum() / 1e3:5.1f}B")


# visualization

def page_data(df: pd.DataFrame) -> dict:
    """column-wise data for the page"""
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


# font (Inter) embedded in the page, so it also works offline
def font_css(fonts_dir: Path) -> str:
    rules = []
    for weight, file in {400: "Inter-Regular.woff2", 600: "Inter-SemiBold.woff2"}.items():
        data = base64.b64encode((fonts_dir / file).read_bytes()).decode()
        rules.append(f"@font-face {{ font-family: 'Inter'; font-weight: {weight}; "
                     f"src: url(data:font/woff2;base64,{data}) format('woff2'); }}")
    return "\n".join(rules)


def write_html(df: pd.DataFrame, template: Path, fonts_dir: Path, out_html: Path) -> None:
    html = (template.read_text(encoding="utf-8")
            .replace("/*FONTS*/", font_css(fonts_dir))
            .replace("/*PLOTLY_JS*/", get_plotlyjs())
            .replace("/*DATA*/null", json.dumps(page_data(df), separators=(",", ":"))))
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    # paths (repository folder = one level above this script)
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    out_csv = root / "data" / "pr_exports.csv"
    out_html = root / "pr_exports.html"

    df = build_dataframe(load_pr_exports(raw_dir), load_world(raw_dir))
    check(df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_html(df, root / "scripts" / "template.html", root / "fonts", out_html)
    print(f"Dataframe: {len(df)} rows -> {out_csv}\nVisualization -> {out_html}")


if __name__ == "__main__":
    main()

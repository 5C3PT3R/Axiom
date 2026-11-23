"""Adapter: UCI Online Retail II (real UK online retailer, Dec 2009 - Dec 2011) -> Axiom's per-day layout.

https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0, no login). Invoice lines become one
completed order per invoice. Real dimensions: country, new_vs_returning (customer's first invoice or not)
and a product category from keywords in the item descriptions. There is no traffic, funnel or payment
data, so conversion metrics are degenerate and payment_method is "n/a" (as with Olist).
"""
from pathlib import Path

import pandas as pd

from axiom.ingestion.olist import write_days

# first keyword hit in the invoice's top-revenue line wins
CATEGORIES = [
    ("christmas", ["CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER"]),
    ("kitchen", ["MUG", "CUP", "PLATE", "BOWL", "TEA", "CAKE", "BAKING", "JAR", "LUNCH", "BOTTLE", "SPOON"]),
    ("lighting", ["LIGHT", "LANTERN", "CANDLE", "T-LIGHT", "LAMP", "HOLDER"]),
    ("bags", ["BAG", "PURSE", "TOTE", "SHOPPER"]),
    ("stationery", ["CARD", "WRAP", "PAPER", "NOTEBOOK", "PENCIL", "STICKER", "TAPE", "ENVELOPE"]),
    ("decor", ["SIGN", "FRAME", "HEART", "CLOCK", "MIRROR", "CUSHION", "DOORMAT", "BUNTING", "GARLAND"]),
    ("toys", ["TOY", "GAME", "DOLL", "PLAYHOUSE", "SPACEBOY", "CHILDREN", "RETROSPOT"]),
]


def _category(desc: str) -> str:
    d = str(desc).upper()
    return next((name for name, kws in CATEGORIES if any(k in d for k in kws)), "other")


def load(src: Path) -> pd.DataFrame:
    """Both yearly sheets of the xlsx as one frame (cached as parquet next to it: the xlsx is slow)."""
    cache = Path(src).with_suffix(".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.concat(pd.read_excel(src, sheet_name=None, dtype={"Invoice": str, "StockCode": str}).values(),
                   ignore_index=True)
    df["Description"] = df.Description.astype(str)  # a few are numbers
    df.to_parquet(cache, index=False)
    return df


def convert(src: Path, out: Path) -> int:
    """src: online_retail_II.xlsx. Writes out/{sessions,events,orders}/<date>.parquet; returns #days."""
    df = load(Path(src))
    df = df[~df.Invoice.str.startswith("C") & (df.Quantity > 0) & (df.Price > 0)].copy()  # no cancellations
    df["line"] = df.Quantity * df.Price
    df["cat"] = df.Description.map(_category)
    top = df.sort_values("line").groupby("Invoice").tail(1).set_index("Invoice").cat
    inv = df.groupby("Invoice").agg(ts=("InvoiceDate", "min"), country=("Country", "first"),
                                    customer=("Customer ID", "first"), amount=("line", "sum"),
                                    items=("StockCode", "count")).reset_index()
    inv["category"] = inv.Invoice.map(top)
    first = inv.dropna(subset=["customer"]).groupby("customer").ts.min()
    inv["new_vs_returning"] = [
        "guest" if pd.isna(c) else "new" if t == first[c] else "returning" for c, t in zip(inv.customer, inv.ts)]
    inv["country"] = inv.country.astype(str)
    inv["order_id"], inv["session_id"] = "o_" + inv.Invoice, "s_" + inv.Invoice
    inv["payment_method"], inv["amount"] = "n/a", inv.amount.round(2)
    return write_days(inv, Path(out))

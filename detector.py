"""Field type inference for SCV Builder."""

import re
import pandas as pd

# UK city → region lookup
CITY_TO_REGION = {
    # Scotland
    "glasgow": "Scotland", "edinburgh": "Scotland", "aberdeen": "Scotland",
    "dundee": "Scotland", "inverness": "Scotland", "perth": "Scotland",
    "stirling": "Scotland", "falkirk": "Scotland", "livingston": "Scotland",
    # Wales
    "cardiff": "Wales", "swansea": "Wales", "newport": "Wales",
    "wrexham": "Wales", "bangor": "Wales", "bridgend": "Wales",
    # Northern Ireland
    "belfast": "Northern Ireland", "londonderry": "Northern Ireland",
    "derry": "Northern Ireland", "lisburn": "Northern Ireland",
    # London
    "london": "London", "city of london": "London",
    # South East
    "brighton": "South East", "southampton": "South East", "portsmouth": "South East",
    "oxford": "South East", "reading": "South East", "guildford": "South East",
    "crawley": "South East", "maidstone": "South East", "slough": "South East",
    "milton keynes": "South East",
    # South West
    "bristol": "South West", "exeter": "South West", "plymouth": "South West",
    "gloucester": "South West", "bath": "South West", "swindon": "South West",
    "bournemouth": "South West", "poole": "South West", "torquay": "South West",
    # East of England
    "cambridge": "East of England", "norwich": "East of England",
    "ipswich": "East of England", "luton": "East of England",
    "peterborough": "East of England", "southend": "East of England",
    "chelmsford": "East of England",
    # East Midlands
    "nottingham": "East Midlands", "leicester": "East Midlands",
    "derby": "East Midlands", "lincoln": "East Midlands",
    "northampton": "East Midlands", "coventry": "East Midlands",
    # West Midlands
    "birmingham": "West Midlands", "wolverhampton": "West Midlands",
    "stoke": "West Midlands", "stoke-on-trent": "West Midlands",
    "walsall": "West Midlands", "west bromwich": "West Midlands",
    "dudley": "West Midlands", "solihull": "West Midlands",
    # Yorkshire and the Humber
    "leeds": "Yorkshire and the Humber", "sheffield": "Yorkshire and the Humber",
    "bradford": "Yorkshire and the Humber", "hull": "Yorkshire and the Humber",
    "york": "Yorkshire and the Humber", "doncaster": "Yorkshire and the Humber",
    "huddersfield": "Yorkshire and the Humber", "wakefield": "Yorkshire and the Humber",
    # North West
    "manchester": "North West", "liverpool": "North West",
    "salford": "North West", "bolton": "North West", "blackpool": "North West",
    "blackburn": "North West", "burnley": "North West", "chester": "North West",
    "carlisle": "North West", "lancaster": "North West", "wigan": "North West",
    "rochdale": "North West", "oldham": "North West", "stockport": "North West",
    # North East
    "newcastle": "North East", "sunderland": "North East",
    "middlesbrough": "North East", "gateshead": "North East",
    "durham": "North East", "hartlepool": "North East",
    # East Midlands extra
    "worcester": "West Midlands", "hereford": "West Midlands",
}

CATEGORICAL_CARDINALITY_THRESHOLD = 50
DATE_NAME_PATTERNS = re.compile(
    r"(date|time|timestamp|dob|birth|created|updated|visited|login|checkout)", re.I
)
GEO_NAME_PATTERNS = re.compile(r"(city|town|location|place|municipality)", re.I)
ID_NAME_PATTERNS = re.compile(r"(_id|id_|_ref|_key|_code)\b", re.I)


def infer_schema(df: pd.DataFrame, source_key: str) -> dict:
    """
    Returns a dict: {col_name: field_type}
    field_type one of: 'customer_id', 'id', 'date', 'geo', 'numeric', 'categorical', 'text'
    """
    schema = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        series = df[col].dropna()

        if col_lower == "customer_id":
            schema[col] = "customer_id"
            continue

        # Try to parse as datetime
        if _is_date_col(col, series):
            schema[col] = "date"
            continue

        if GEO_NAME_PATTERNS.search(col_lower):
            schema[col] = "geo"
            continue

        if ID_NAME_PATTERNS.search(col_lower) and series.nunique() / max(len(series), 1) > 0.3:
            schema[col] = "id"
            continue

        if pd.api.types.is_numeric_dtype(series):
            schema[col] = "numeric"
            continue

        n_unique = series.nunique()
        if n_unique <= CATEGORICAL_CARDINALITY_THRESHOLD:
            schema[col] = "categorical"
        else:
            schema[col] = "text"

    return schema


def _is_date_col(col: str, series: pd.Series) -> bool:
    if DATE_NAME_PATTERNS.search(col):
        return True
    if series.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        sample = series.dropna().head(20)
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.7
    return False


def city_to_region(city_val) -> str:
    if pd.isna(city_val):
        return "Unknown"
    return CITY_TO_REGION.get(str(city_val).lower().strip(), "Other")

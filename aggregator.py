"""Aggregation engine for SCV Builder."""

import re
import pandas as pd
import numpy as np
from detector import infer_schema, city_to_region

AGE_BINS = [0, 17, 24, 34, 44, 54, 64, 74, 200]
AGE_LABELS = ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75+"]
CATEGORICAL_FLAG_MAX_UNIQUE = 5


def _is_customer_level(df):
    """True if Customer_ID is already unique in this sheet — no aggregation needed."""
    return df["Customer_ID"].nunique() == len(df)


def build_scv(sheets, reference_date=None):
    """
    sheets: list of {name, prefix, df}
    Returns (scv_df, column_specs)
    column_specs: [{"column": str, "source": str, "derivation": str, "include": bool}]
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    per_sheet_frames = []
    column_specs = []

    for sheet in sheets:
        name = sheet["name"]
        prefix = sheet["prefix"].rstrip("_") + "_"
        df = sheet["df"].copy()

        if "Customer_ID" not in df.columns:
            continue

        schema = infer_schema(df, name)
        customer_level = _is_customer_level(df)
        agg_df, specs = _aggregate_sheet(df, schema, prefix, name, reference_date, customer_level)
        per_sheet_frames.append(agg_df)
        column_specs.extend(specs)

    if not per_sheet_frames:
        return pd.DataFrame(), []

    # Merge all sheets on Customer_ID
    scv = per_sheet_frames[0]
    for frame in per_sheet_frames[1:]:
        scv = scv.merge(frame, on="Customer_ID", how="outer")

    # Resolve conflicts: columns duplicated across sheets (suffixed _x, _y) — keep most recent (last non-null)
    scv = _resolve_conflicts(scv)

    # Mark all column_specs as include=True by default
    for spec in column_specs:
        spec.setdefault("include", True)

    return scv, column_specs


def _aggregate_sheet(df, schema, prefix, source, reference_date, customer_level=False):
    agg = {}
    specs = []
    cid = "Customer_ID"

    for col, ftype in schema.items():
        if ftype == "customer_id":
            continue
        col_data = df[[cid, col]].copy()

        # ── Customer-level sheet: carry columns as-is, only derive from dates ──
        if customer_level:
            if ftype == "date":
                # fall through to date handling below
                pass
            elif ftype == "numeric":
                col_clean = _clean_name(col)
                out_col = f"{prefix}{col_clean}"
                agg[out_col] = col_data.set_index(cid)[col]
                specs.append({"column": out_col, "source": source, "derivation": f"{col} (as-is)"})
                # Bucketing only — no aggregation stats
                if re.search(r"(salary|income|spend|revenue|value|amount|age)", col, re.I):
                    bucket_col = f"{prefix}{col_clean}_Band"
                    try:
                        banded = pd.qcut(col_data[col], q=3, labels=["Low", "Medium", "High"], duplicates="drop")
                    except Exception:
                        banded = pd.Series("Unknown", index=col_data.index)
                    banded.index = col_data[cid]
                    agg[bucket_col] = banded
                    specs.append({"column": bucket_col, "source": source, "derivation": f"Percentile band of {col}"})
                continue
            else:
                col_clean = _clean_name(col)
                out_col = f"{prefix}{col_clean}"
                agg[out_col] = col_data.set_index(cid)[col]
                specs.append({"column": out_col, "source": source, "derivation": f"{col} (as-is)"})
                continue

        # ── Transaction-level sheet: full aggregation ──
        if ftype == "id":
            out_col = f"{prefix}Count_{col}"
            agg[out_col] = col_data.groupby(cid)[col].nunique()
            specs.append({"column": out_col, "source": source, "derivation": f"Count distinct {col}"})

        elif ftype == "numeric":
            col_data[col] = pd.to_numeric(col_data[col], errors="coerce")
            grp = col_data.groupby(cid)[col]
            col_clean = _clean_name(col)
            for stat, series in [
                (f"{prefix}Avg_{col_clean}", grp.mean()),
                (f"{prefix}Sum_{col_clean}", grp.sum()),
                (f"{prefix}Min_{col_clean}", grp.min()),
                (f"{prefix}Max_{col_clean}", grp.max()),
            ]:
                agg[stat] = series
                specs.append({"column": stat, "source": source, "derivation": f"{stat.split('_')[1]} of {col}"})

            if re.search(r"(salary|income|spend|revenue|value|amount)", col, re.I):
                bucket_col = f"{prefix}{col_clean}_Band"
                try:
                    banded = pd.qcut(grp.mean(), q=3, labels=["Low", "Medium", "High"], duplicates="drop")
                except Exception:
                    banded = pd.Series("Unknown", index=grp.mean().index)
                agg[bucket_col] = banded
                specs.append({"column": bucket_col, "source": source, "derivation": f"Percentile band of {col}"})

        elif ftype == "categorical":
            col_clean = _clean_name(col)
            grp = col_data.groupby(cid)[col]
            n_unique = col_data[col].dropna().nunique()

            count_distinct_col = f"{prefix}Count_Distinct_{col_clean}"
            agg[count_distinct_col] = grp.nunique()
            specs.append({"column": count_distinct_col, "source": source,
                          "derivation": f"Count distinct {col}"})

            if n_unique <= CATEGORICAL_FLAG_MAX_UNIQUE:
                for val in sorted(str(v) for v in col_data[col].dropna().unique()):
                    flag_col = f"{prefix}Has_{col_clean}_{_clean_name(val)}"
                    flag = grp.apply(lambda s, v=val: (s.astype(str) == v).any()).astype(int)
                    agg[flag_col] = flag
                    specs.append({"column": flag_col, "source": source, "derivation": f"Flag: {col}={val}"})

        elif ftype == "geo":
            col_clean = _clean_name(col)
            region_col = f"{prefix}Region"
            # Most frequent region per customer
            region_series = col_data.copy()
            region_series["_region"] = region_series[col].apply(city_to_region)
            agg[region_col] = region_series.groupby(cid)["_region"].agg(
                lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"
            )
            specs.append({"column": region_col, "source": source, "derivation": f"UK Region from {col}"})

        elif ftype == "date":
            col_clean = _clean_name(col)
            col_data[col] = pd.to_datetime(col_data[col], errors="coerce")

            dob_like = bool(re.search(r"(dob|birth)", col, re.I))
            visit_like = bool(re.search(r"(visit|session|login|view)", col, re.I))

            if customer_level:
                # For already-aggregated sheets: derive from the single date value per customer
                date_series = col_data.set_index(cid)[col]
                if dob_like:
                    age_col = f"{prefix}Age_Years"
                    agg[age_col] = ((reference_date - date_series).dt.days / 365.25).round(1)
                    specs.append({"column": age_col, "source": source, "derivation": f"Age in years from {col}"})
                    age_band_col = f"{prefix}Age_Band"
                    agg[age_band_col] = pd.cut(agg[age_col], bins=AGE_BINS, labels=AGE_LABELS, right=True)
                    specs.append({"column": age_band_col, "source": source, "derivation": "Age band"})
                else:
                    last_col = f"{prefix}Days_Since_{col_clean}"
                    agg[last_col] = (reference_date - date_series).dt.days
                    specs.append({"column": last_col, "source": source, "derivation": f"Days since {col}"})
                continue

            grp = col_data.groupby(cid)[col]

            if dob_like:
                age_col = f"{prefix}Age_Years"
                agg[age_col] = ((reference_date - grp.max()).dt.days / 365.25).round(1)
                specs.append({"column": age_col, "source": source, "derivation": f"Age in years from {col}"})
                age_band_col = f"{prefix}Age_Band"
                agg[age_band_col] = pd.cut(agg[age_col], bins=AGE_BINS, labels=AGE_LABELS, right=True)
                specs.append({"column": age_band_col, "source": source, "derivation": "Age band"})

            elif visit_like:
                active_days_col = f"{prefix}Active_Days"
                visits_col = f"{prefix}Visit_Count"
                morning_col = f"{prefix}Visits_Morning"
                evening_col = f"{prefix}Visits_Evening"
                night_col = f"{prefix}Visits_Night"

                agg[active_days_col] = grp.apply(lambda s: s.dropna().dt.date.nunique())
                agg[visits_col] = grp.count()
                specs.append({"column": active_days_col, "source": source, "derivation": "Distinct active days"})
                specs.append({"column": visits_col, "source": source, "derivation": "Total visit count"})

                hour_data = col_data.copy()
                hour_data["_hour"] = col_data[col].dt.hour
                hour_grp = hour_data.groupby(cid)["_hour"]
                agg[morning_col] = hour_grp.apply(lambda h: ((h >= 6) & (h < 12)).sum())
                agg[evening_col] = hour_grp.apply(lambda h: ((h >= 18) & (h < 22)).sum())
                agg[night_col] = hour_grp.apply(lambda h: ((h >= 22) | (h < 6)).sum())
                for c, d in [(morning_col, "Visits 06:00–11:59"),
                             (evening_col, "Visits 18:00–21:59"),
                             (night_col, "Visits 22:00–05:59")]:
                    specs.append({"column": c, "source": source, "derivation": d})

            else:
                last_col = f"{prefix}Days_Since_Last_{col_clean}"
                first_col = f"{prefix}Tenure_Days_{col_clean}"
                agg[last_col] = (reference_date - grp.max()).dt.days
                agg[first_col] = (reference_date - grp.min()).dt.days
                specs.append({"column": last_col, "source": source, "derivation": f"Days since last {col}"})
                specs.append({"column": first_col, "source": source, "derivation": f"Tenure days since first {col}"})

        elif ftype == "text":
            # Carry most frequent value
            col_clean = _clean_name(col)
            out_col = f"{prefix}{col_clean}"
            agg[out_col] = col_data.groupby(cid)[col].agg(
                lambda x: x.dropna().mode().iloc[0] if x.dropna().shape[0] > 0 else None
            )
            specs.append({"column": out_col, "source": source, "derivation": f"Most frequent {col}"})

    # Combine aggregates
    result = pd.DataFrame({"Customer_ID": df["Customer_ID"].unique()}).set_index("Customer_ID")
    for col_name, series in agg.items():
        result[col_name] = series
    result = result.reset_index()
    return result, specs


def _resolve_conflicts(df):
    """Drop _x/_y duplicates by keeping the non-null value (prefer _x)."""
    cols_x = [c for c in df.columns if c.endswith("_x")]
    for cx in cols_x:
        base = cx[:-2]
        cy = base + "_y"
        if cy in df.columns:
            df[base] = df[cx].combine_first(df[cy])
            df.drop(columns=[cx, cy], inplace=True)
    return df


def _clean_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(s)).strip("_")


def apply_column_selection(df, column_specs):
    """Filter df to only included columns."""
    keep = {"Customer_ID"} | {s["column"] for s in column_specs if s.get("include", True)}
    cols = [c for c in df.columns if c in keep]
    return df[cols]

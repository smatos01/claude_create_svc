"""SCV Builder — Single Customer View builder from multi-sheet Excel files."""

import io
import pandas as pd
import streamlit as st
from aggregator import build_scv, apply_column_selection
from exporter import to_excel_bytes

st.set_page_config(page_title="SCV Builder", page_icon="🧩", layout="wide")

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .title-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
  }
  .title-banner h1 { margin: 0; font-size: 2.2rem; }
  .title-banner p  { margin: 0.4rem 0 0; opacity: 0.8; font-size: 1rem; }
  .step-badge {
    display: inline-block;
    background: #0f3460;
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    font-weight: bold;
    margin-right: 8px;
  }
  .spec-table th { background: #f0f4ff; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-banner">
  <h1>🧩 SCV Builder</h1>
  <p>Combine multiple data sources into one unified, customer-level Single Customer View</p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
for key in ("sheets_config", "scv_df", "column_specs", "step"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "step" else 1

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown("### <span class='step-badge'>1</span> Upload your Excel file", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Each sheet will be treated as a separate data source. All sheets must contain a **Customer_ID** column.",
    type=["xlsx", "xls"],
    label_visibility="visible",
)

if uploaded:
    try:
        xl = pd.ExcelFile(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read the Excel file: {e}")
        st.stop()

    sheet_names = xl.sheet_names
    if not sheet_names:
        st.warning("The uploaded file has no sheets.")
        st.stop()

    # ── Step 2: Configure sheets ───────────────────────────────────────────────
    st.markdown("### <span class='step-badge'>2</span> Configure data sources", unsafe_allow_html=True)
    st.caption("Assign a prefix for each sheet. Columns in the SCV will be prefixed accordingly.")

    DEFAULT_PREFIXES = ["TXN_", "CRM_", "WEB_", "PROD_", "SRVC_", "MKTG_"]

    sheets_config = []
    cols = st.columns(min(len(sheet_names), 3))
    for i, sname in enumerate(sheet_names):
        with cols[i % len(cols)]:
            st.markdown(f"**{sname}**")
            default_prefix = DEFAULT_PREFIXES[i] if i < len(DEFAULT_PREFIXES) else f"SRC{i+1}_"
            prefix = st.text_input(
                f"Prefix for '{sname}'",
                value=default_prefix,
                key=f"prefix_{sname}",
                label_visibility="collapsed",
                placeholder="e.g. TXN_",
            )
            include = st.checkbox(f"Include sheet", value=True, key=f"include_{sname}")
            if include:
                sheets_config.append({"name": sname, "prefix": prefix})

    if not sheets_config:
        st.warning("Please include at least one sheet.")
        st.stop()

    # Reference date for age / recency calculations
    ref_date = st.date_input(
        "Reference date (used for recency / age calculations)",
        value=pd.Timestamp.now().date(),
    )

    if st.button("▶ Build SCV preview", type="primary"):
        dfs = []
        errors = []
        for cfg in sheets_config:
            try:
                df = xl.parse(cfg["name"])
                df.columns = [str(c).strip() for c in df.columns]
                if "Customer_ID" not in df.columns:
                    errors.append(f"Sheet **{cfg['name']}** has no Customer_ID column — skipped.")
                else:
                    dfs.append({**cfg, "df": df})
            except Exception as e:
                errors.append(f"Error reading sheet **{cfg['name']}**: {e}")

        for err in errors:
            st.warning(err)

        if not dfs:
            st.error("No valid sheets to process.")
            st.stop()

        with st.spinner("Building SCV…"):
            scv_df, column_specs = build_scv(dfs, reference_date=pd.Timestamp(ref_date))

        st.session_state.scv_df = scv_df
        st.session_state.column_specs = column_specs
        st.session_state.step = 3

# ── Step 3: Schema preview ─────────────────────────────────────────────────────
if st.session_state.step >= 3 and st.session_state.column_specs is not None:
    st.markdown("### <span class='step-badge'>3</span> Review output schema", unsafe_allow_html=True)
    st.caption(
        "The table below shows every column that will appear in the SCV. "
        "Un-tick any columns you want to exclude before downloading."
    )

    specs = st.session_state.column_specs

    # Group by source for a cleaner UI
    sources = sorted({s["source"] for s in specs})
    updated_specs = []

    for src in sources:
        src_specs = [s for s in specs if s["source"] == src]
        with st.expander(f"**{src}** — {len(src_specs)} columns", expanded=True):
            col_a, col_b, col_c = st.columns([3, 4, 1])
            col_a.markdown("**Column name**")
            col_b.markdown("**Derivation**")
            col_c.markdown("**Include**")
            for spec in src_specs:
                ca, cb, cc = st.columns([3, 4, 1])
                ca.code(spec["column"], language=None)
                cb.caption(spec["derivation"])
                included = cc.checkbox(
                    "include",
                    value=spec.get("include", True),
                    key=f"inc_{spec['column']}",
                    label_visibility="collapsed",
                )
                updated_specs.append({**spec, "include": included})

    st.session_state.column_specs = updated_specs

    included_count = sum(1 for s in updated_specs if s.get("include", True))
    scv_rows = len(st.session_state.scv_df)
    st.info(
        f"Output: **{scv_rows:,} customers** × **{included_count + 1} columns** "
        f"(including Customer_ID)"
    )

    # ── Step 4: Preview & download ─────────────────────────────────────────────
    st.markdown("### <span class='step-badge'>4</span> Preview & download", unsafe_allow_html=True)

    final_df = apply_column_selection(
        st.session_state.scv_df, st.session_state.column_specs
    )

    st.dataframe(final_df.head(100), use_container_width=True, height=350)
    if len(final_df) > 100:
        st.caption(f"Showing first 100 of {len(final_df):,} rows.")

    excel_bytes = to_excel_bytes(final_df)
    st.download_button(
        label="⬇ Download SCV as Excel",
        data=excel_bytes,
        file_name="single_customer_view.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

"""Excel export for SCV Builder."""

import io
import pandas as pd


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="SCV")
        ws = writer.sheets["SCV"]

        # Freeze header row
        ws.freeze_panes = "A2"

        # Auto-size columns (cap at 50)
        for col_cells in ws.columns:
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 50)

    return buf.getvalue()

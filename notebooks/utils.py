from curses import raw
import pandas as pd
import re
from pathlib import Path

def promote_first_row_to_header(df):
    try:
        unnamed = any(str(c).startswith("Unnamed") for c in df.columns)
    except Exception:
        unnamed = False
    if unnamed or df.columns.duplicated().any():
        first_row = df.iloc[0].fillna("").astype(str).str.strip().tolist()
        if any(first_row):
            df = df.iloc[1:].copy()
            df.columns = first_row
            df = df.reset_index(drop=True)
    return df


def clean_nfl_string(text, sep='-', keep_left=True):
    if not isinstance(text, str):
        return None
    parts = text.split(sep)
    if len(parts) < 2:
        return text
    result = parts[0] if keep_left else parts[1]
    return result.strip()
import numpy as np, io
from typing import List, Dict, Tuple, Optional, Any
import json
import pandas as pd

def _df_to_records_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to a JSON-safe list[dict] (NaNs -> None)."""
    if df is None or df.empty:
        return []
    # Ensure plain Python scalars (not numpy types) and replace NaN with None
    out = df.replace({np.nan: None}).to_dict(orient="records")
    return out

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    buf = io.StringIO()
    df.replace({np.nan: ""}).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

#part 0: utils
BASE_TABLE = np.full(256, -1, dtype=np.int8)
BASE_TABLE[ord("A")] = 0
BASE_TABLE[ord("C")] = 1
BASE_TABLE[ord("G")] = 2
BASE_TABLE[ord("T")] = 3

def one_hot_encode(seq: str) -> np.ndarray:
    # Turn seq into a 1D array of uint8 (ASCII codes), then map through BASE_TABLE
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)  # shape = (N,)
    idxs = BASE_TABLE[raw]  # shape = (N,), e.g. [0, 2, -1, 3, …]
    N = idxs.shape[0]
    onehot = np.zeros((N, 4), dtype=np.float32)
    valid = idxs >= 0
    onehot[np.nonzero(valid), idxs[valid]] = 1.0
    return onehot
def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTRYMKSWHBVDNacgtrymkswhbvdn", "TGCAYRKMSWDVBHNtgcayrkmswdvbhn")
    return seq.translate(comp)[::-1]
def iupac_to_motif(iupac: str) -> np.ndarray:
    scoring = {
        "A" : [1, 0, 0, 0],
        "C" : [0, 1, 0, 0],
        "G" : [0, 0, 1, 0],
        "T" : [0, 0, 0, 1],
        "R" : [0.5, 0, 0.5, 0],
        "Y" : [0, 0.5, 0, 0.5],
        "S" : [0, 0.5, 0.5, 0],
        "W" : [0.5, 0, 0, 0.5],
        "K" : [0, 0, 0.5, 0.5],
        "M" : [0.5, 0.5, 0, 0],
        "B" : [0, 1/3, 1/3, 1/3],
        "D" : [1/3, 0, 1/3, 1/3],
        "H" : [1/3, 1/3, 0, 1/3],
        "V" : [1/3, 1/3, 1/3, 0],
        "N" : [0.25, 0.25, 0.25, 0.25],
    }
    pwm = np.zeros((len(iupac), 4))
    for i in range (len(iupac)):
        pwm[i] = scoring[iupac[i]]
    return pwm

def compute_max_motif_score(
    motif_hits: Dict[str, Dict[str, List[Tuple[str,int,int,int,float]]]]
) -> float:
    all_scores = (
        score
        for by_peak in motif_hits.values()
        for hits in by_peak.values()
        for (_c, _s, _e, _d, score) in hits
    )
    return max(all_scores, default=0.0)


def _parse_per_motif_pvals(raw: Optional[str]) -> Dict[str, float]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        out = {}
        for k, v in obj.items():
            try:
                fv = float(v)
                if fv >= 0:   # leave validation simple; UI can enforce upper bounds
                    out[str(k)] = fv
            except Exception:
                continue
        return out
    except Exception:
        return {}
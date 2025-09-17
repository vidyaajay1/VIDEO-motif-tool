import os, json, io, pickle, zipfile
import pandas as pd
from rq import get_current_job
from typing import List, Tuple
from app.new_scan_motifs import scan_wrapper
from app.plotting import plot_occurrence_overview_plotly, rank_peaks_for_plot 
from app.utils import _parse_per_motif_pvals, _df_to_csv_bytes, _df_to_records_json_safe  # adjust
from app.models import DatasetInfo, BatchScanResponse
from fastapi import HTTPException
from app.filter_motifs import filter_motif_hits
from app.integrated_scoring import score_and_merge, score_hit_naive_bayes
from app.bigwig_overlay import fetch_bw_coverage, plot_chip_overlay

def _load_session(session_id: str) -> List[DatasetInfo]:
    try:
        with open(os.path.join(TMP_DIR, f"{session_id}_datasets.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown comparison session_id")

TMP_DIR = "tmp"

def _progress(pct: int):
    """Helper to update job progress in RQ meta."""
    job = get_current_job()
    if job:
        job.meta["progress"] = pct
        job.save_meta()

def _ensure(p: str, label: str | None = None):
    if not os.path.exists(p):
        who = f" for {label}" if label else ""
        raise RuntimeError(f"Missing cache{who}: {os.path.basename(p)}")
    

def _write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)

def scan_single_dataset(data_id: str, window: int, fimo_threshold: float) -> dict:
    """
    Heavy CPU task: run scan_wrapper for one dataset and write the same cache files as today.
    Return a small payload (no big arrays!) so Redis stays lean.
    """
    _progress(1)

    try:
        motif_list = pd.read_pickle(f"{TMP_DIR}/{data_id}_motif_list.pkl")
    except Exception as e:
        raise RuntimeError(f"Motif list missing/invalid for {data_id}: {e}")

    try:
        peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
    except FileNotFoundError:
        raise RuntimeError(f"No cached peak data for {data_id}. Run Process Genomic Input first.")

    _progress(10)

    motif_hits, df_hits = scan_wrapper(
        peaks_df,
        "fasta/genome.fa",
        window,
        motif_list,
        fimo_threshold
    )

    _progress(90)

    with open(f"{TMP_DIR}/{data_id}_motif_hits.pkl", "wb") as f:
        pickle.dump(motif_hits, f)
    with open(f"{TMP_DIR}/{data_id}_df_hits.pkl", "wb") as f:
        pickle.dump(df_hits, f)

    _progress(100)
    return {"data_id": data_id, "ok": True}

def scan_session_batch(session_id: str, dataset_ids: list[str], window: int, fimo_threshold: float) -> dict:
    """
    Heavy CPU task for compare-mode: loop datasets for a session.
    Writes per-dataset cache files as before.
    """
    # load session-level motif list
    try:
        motif_list = pd.read_pickle(f"{TMP_DIR}/{session_id}_motif_list.pkl")
    except FileNotFoundError:
        raise RuntimeError(f"Run /validate-motifs-group first for session {session_id}")

    completed = []
    total = max(1, len(dataset_ids))
    for i, data_id in enumerate(dataset_ids, start=1):
        # get peaks
        try:
            peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
        except FileNotFoundError:
            raise RuntimeError(f"No cached peaks for {data_id}")

        motif_hits, df_hits = scan_wrapper(
            peaks_df,
            "fasta/genome.fa",
            window,
            motif_list,
            fimo_threshold
        )
        with open(f"{TMP_DIR}/{data_id}_motif_hits.pkl", "wb") as f:
            pickle.dump(motif_hits, f)
        with open(f"{TMP_DIR}/{data_id}_df_hits.pkl", "wb") as f:
            pickle.dump(df_hits, f)

        completed.append(data_id)
        # progress
        pct = int((i / total) * 100)
        _progress(min(99, pct))

    _progress(100)
    return {"session_id": session_id, "datasets": completed, "ok": True}


# --------------------------
# 1) FILTER + SCORE (single)
# --------------------------
def filter_score_single_task(data_id: str, have_atac: bool, have_chip: bool) -> dict:
    _progress(1)
    df_hits_fp = os.path.join(TMP_DIR, f"{data_id}_df_hits.pkl")
    _ensure(df_hits_fp)
    with open(df_hits_fp, "rb") as f:
        df_hits = pickle.load(f)

    empty_hits = pd.DataFrame(columns=df_hits.columns)

    # load uploaded BEDs if present (web wrote these before enqueuing)
    atac_path = os.path.join(TMP_DIR, f"{data_id}_atac.bed")
    chip_path = os.path.join(TMP_DIR, f"{data_id}_chip.bed")

    # ATAC
    if have_atac and os.path.exists(atac_path):
        atac_filt = filter_motif_hits(df_hits, atac_path)
    else:
        atac_filt = empty_hits
    with open(os.path.join(TMP_DIR, f"{data_id}_atac_filt_hits.pkl"), "wb") as f:
        pickle.dump(atac_filt, f)

    # ChIP
    if have_chip and os.path.exists(chip_path):
        chip_filt = filter_motif_hits(df_hits, chip_path)
    else:
        chip_filt = empty_hits
    with open(os.path.join(TMP_DIR, f"{data_id}_chip_filt_hits.pkl"), "wb") as f:
        pickle.dump(chip_filt, f)

    _progress(50)
    scored = score_and_merge(df_hits, chip_filt, atac_filt)
    ranked = score_hit_naive_bayes(scored)
    top_hits = (ranked.reset_index()
                        .sort_values("P_regulatory", ascending=False)
                        [["Peak_ID","Motif","P_regulatory","M_prom","M_chip","M_atac","logFC","FIMO_score"]])
    top_fp = os.path.join(TMP_DIR, f"{data_id}_top_hits.tsv")
    top_hits.to_csv(top_fp, sep="\t", index=False)

    _progress(100)
    return {"data_id": data_id, "ok": True}

# ------------------------------
# 2) FILTER + SCORE (compare/batch)
# ------------------------------
def filter_score_batch_task(session_id: str, mode: str) -> dict:
    """
    mode:
      - 'shared' : use shared *_atac_shared.bed / *_chip_shared.bed for all datasets
      - 'ab'     : use per-dataset A/B uploads saved as {data_id}_atac.bed / _chip.bed
    """
    datasets = _load_session(session_id)
    if not datasets:
        raise RuntimeError("Empty/unknown session")

    # Preload shared bytes paths if present
    atac_shared = os.path.join(TMP_DIR, f"{session_id}_atac_shared.bed")
    chip_shared = os.path.join(TMP_DIR, f"{session_id}_chip_shared.bed")
    have_atac_shared = os.path.exists(atac_shared)
    have_chip_shared = os.path.exists(chip_shared)

    processed = []
    total = max(1, len(datasets))

    for i, ds in enumerate(datasets, start=1):
        data_id = ds.data_id
        df_hits_fp = os.path.join(TMP_DIR, f"{data_id}_df_hits.pkl")
        _ensure(df_hits_fp, ds.label)
        with open(df_hits_fp, "rb") as f:
            df_hits = pickle.load(f)
        empty_hits = pd.DataFrame(columns=df_hits.columns)

        # Decide which beds to use
        if mode == "shared":
            atac_path = atac_shared if have_atac_shared else None
            chip_path = chip_shared if have_chip_shared else None
        else:  # 'ab'
            atac_cand = os.path.join(TMP_DIR, f"{data_id}_atac.bed")
            chip_cand = os.path.join(TMP_DIR, f"{data_id}_chip.bed")
            atac_path = atac_cand if os.path.exists(atac_cand) else None
            chip_path = chip_cand if os.path.exists(chip_cand) else None

        # ATAC
        if atac_path:
            atac_filt = filter_motif_hits(df_hits, atac_path)
        else:
            atac_filt = empty_hits
        with open(os.path.join(TMP_DIR, f"{data_id}_atac_filt_hits.pkl"), "wb") as f:
            pickle.dump(atac_filt, f)

        # ChIP
        if chip_path:
            chip_filt = filter_motif_hits(df_hits, chip_path)
        else:
            chip_filt = empty_hits
        with open(os.path.join(TMP_DIR, f"{data_id}_chip_filt_hits.pkl"), "wb") as f:
            pickle.dump(chip_filt, f)

        # Score + export top hits
        scored = score_and_merge(df_hits, chip_filt, atac_filt)
        ranked = score_hit_naive_bayes(scored)
        top_hits = (ranked.reset_index()
                            .sort_values("P_regulatory", ascending=False)
                            [["Peak_ID","Motif","P_regulatory","M_prom","M_chip","M_atac","logFC","FIMO_score"]])
        top_fp = os.path.join(TMP_DIR, f"{data_id}_top_hits.tsv")
        top_hits.to_csv(top_fp, sep="\t", index=False)

        processed.append(data_id)
        _progress(min(99, int(i/total*100)))

    _progress(100)
    return {"session_id": session_id, "datasets": processed, "ok": True}

# -------------------------------------------------
# 3) FILTERED OVERVIEW PLOTS (single & compare)
# -------------------------------------------------
def plot_filtered_single_task(
    data_id: str,
    chip: bool,
    atac: bool,
    window: int,
    use_hit_number: bool,
    use_match_score: bool,
    chosen_motif: str | None,
    best_transcript: bool,
    per_motif_pvals_json: str,
    want_download: bool,
):
    _progress(1)
    peaks_fp   = os.path.join(TMP_DIR, f"{data_id}_peaks.pkl")
    gene_fp    = os.path.join(TMP_DIR, f"{data_id}_genes_lfc.pkl")
    motifs_fp  = os.path.join(TMP_DIR, f"{data_id}_motif_list.pkl")
    _ensure(peaks_fp); _ensure(gene_fp); _ensure(motifs_fp)

    peaks_df   = pd.read_pickle(peaks_fp)
    gene_lfc   = pd.read_pickle(gene_fp)
    motif_list = pd.read_pickle(motifs_fp)

    # decide df_to_plot
    if not (chip or atac):
        df_to_plot = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_df_hits.pkl"))
    else:
        dfs = []
        if chip and os.path.exists(os.path.join(TMP_DIR, f"{data_id}_chip_filt_hits.pkl")):
            dfs.append(pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_chip_filt_hits.pkl")))
        if atac and os.path.exists(os.path.join(TMP_DIR, f"{data_id}_atac_filt_hits.pkl")):
            dfs.append(pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_atac_filt_hits.pkl")))
        if not dfs:
            raise RuntimeError("Requested filtered plots but no filtered hits found.")
        df_to_plot = dfs[0] if len(dfs)==1 else pd.merge(dfs[0], dfs[1], how="inner")

    _progress(20)
    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)

    peak_ranks = rank_peaks_for_plot(
        df_hits=df_to_plot,
        gene_lfc=gene_lfc,
        peaks_df=peaks_df,
        use_hit_number=use_hit_number,
        use_match_score=use_match_score,
        motif=(chosen_motif or None),
        best_transcript=best_transcript,
        min_score_bits=0.0,
        per_motif_pvals=per_motif_pvals,
    )

    fig, ordered_peaks, final_hits_df = plot_occurrence_overview_plotly(
        peak_list=list(peak_ranks.keys()),
        peaks_df=peaks_df,
        motif_list=motif_list,
        df_hits=df_to_plot,
        window=window,
        peak_rank=peak_ranks,
        title="Motif hits",
        min_score_bits=0.0,
        per_motif_pvals=per_motif_pvals,
    )

    _write_json(os.path.join(TMP_DIR, f"{data_id}_filtered_overview.json"), json.loads(fig.to_json()))
    _write_json(os.path.join(TMP_DIR, f"{data_id}_filtered_ordered.json"), ordered_peaks)
    _write_json(os.path.join(TMP_DIR, f"{data_id}_filtered_final_hits.json"), _df_to_records_json_safe(final_hits_df))

    if want_download:
        csv_bytes = _df_to_csv_bytes(final_hits_df)
        with open(os.path.join(TMP_DIR, f"{data_id}_filtered_final_hits.csv"), "wb") as f:
            f.write(csv_bytes)

    _progress(100)
    return {"data_id": data_id, "ok": True}

def plot_filtered_compare_task(
    session_id: str,
    window: int,
    chip: bool,
    atac: bool,
    use_hit_number: bool,
    use_match_score: bool,
    chosen_motif: str | None,
    best_transcript: bool,
    per_motif_pvals_json: str,
    want_download: bool,
    merge_files: bool,
):
    _progress(1)
    datasets = _load_session(session_id)
    motif_fp = os.path.join(TMP_DIR, f"{session_id}_motif_list.pkl")
    _ensure(motif_fp)
    motif_list = pd.read_pickle(motif_fp)
    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)

    dfs = {}
    total = max(1, len(datasets))
    for i, ds in enumerate(datasets, start=1):
        data_id, label = ds.data_id, ds.label
        peaks_df = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_peaks.pkl"))
        gene_lfc = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_genes_lfc.pkl"))

        if not (chip or atac):
            df_to_plot = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_df_hits.pkl"))
        else:
            parts = []
            if chip and os.path.exists(os.path.join(TMP_DIR, f"{data_id}_chip_filt_hits.pkl")):
                parts.append(pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_chip_filt_hits.pkl")))
            if atac and os.path.exists(os.path.join(TMP_DIR, f"{data_id}_atac_filt_hits.pkl")):
                parts.append(pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_atac_filt_hits.pkl")))
            if not parts:
                raise RuntimeError(f"No filtered hits available for {label}")
            df_to_plot = parts[0] if len(parts)==1 else pd.merge(parts[0], parts[1], how="inner")

        peak_ranks = rank_peaks_for_plot(
            df_hits=df_to_plot,
            gene_lfc=gene_lfc,
            peaks_df=peaks_df,
            use_hit_number=use_hit_number,
            use_match_score=use_match_score,
            motif=(chosen_motif or None),
            best_transcript=best_transcript,
            min_score_bits=0.0,
            per_motif_pvals=per_motif_pvals,
        )

        fig, ordered_peaks, final_hits_df = plot_occurrence_overview_plotly(
            peak_list=list(peak_ranks.keys()),
            peaks_df=peaks_df,
            motif_list=motif_list,
            df_hits=df_to_plot,
            window=window,
            peak_rank=peak_ranks,
            title=f"Motif hits - {label}",
            min_score_bits=0.0,
            per_motif_pvals=per_motif_pvals,
        )

        _write_json(os.path.join(TMP_DIR, f"{session_id}_{label}_filtered_overview.json"), json.loads(fig.to_json()))
        _write_json(os.path.join(TMP_DIR, f"{session_id}_{label}_filtered_ordered.json"), ordered_peaks)
        _write_json(os.path.join(TMP_DIR, f"{session_id}_{label}_filtered_final_hits.json"), _df_to_records_json_safe(final_hits_df))
        dfs[label] = final_hits_df

        _progress(min(99, int(i/total*100)))

    if want_download:
        if merge_files:
            merged = []
            for label, df in dfs.items():
                x = df.copy()
                x.insert(0, "Dataset", label)
                merged.append(x)
            merged_df = pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()
            with open(os.path.join(TMP_DIR, f"{session_id}_filtered_final_hits_merged.csv"), "wb") as f:
                f.write(_df_to_csv_bytes(merged_df))
        else:
            zip_path = os.path.join(TMP_DIR, f"{session_id}_filtered_final_hits.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for label, df in dfs.items():
                    z.writestr(f"{label}_filtered_final_hits.csv", _df_to_csv_bytes(df))

    _progress(100)
    return {"session_id": session_id, "ok": True}

def chip_overlay_single_task(
    data_id: str,
    gene: str,
    window: int,
    bw_inputs: List[Tuple[str, str, str]],  # [(path, label, color)]
    per_motif_pvals_json: str,
    min_score_bits: float,
) -> dict:
    _progress(1)
    peaks_fp = os.path.join(TMP_DIR, f"{data_id}_peaks.pkl")
    motifs_fp = os.path.join(TMP_DIR, f"{data_id}_motif_list.pkl")
    hits_fp = os.path.join(TMP_DIR, f"{data_id}_df_hits.pkl")
    _ensure(peaks_fp, "peaks"); _ensure(hits_fp, "df_hits"); _ensure(motifs_fp, "motif_list")

    peaks_df = pd.read_pickle(peaks_fp)
    df_hits = pd.read_pickle(hits_fp)
    with open(motifs_fp, "rb") as f:
        motif_list = pickle.load(f)

    _progress(10)
    coverage_data = fetch_bw_coverage(str(gene), peaks_df, bw_inputs)
    _progress(50)

    if not coverage_data:
        raise RuntimeError(f"No coverage data for {gene}")

    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)

    fig = plot_chip_overlay(
        coverage_data=coverage_data,
        motif_list=motif_list,
        df_hits=df_hits,
        peaks_df=peaks_df,
        peak_id=str(gene),
        window=int(window),
        per_motif_pvals=per_motif_pvals,
        min_score_bits=float(min_score_bits),
        title=f"{gene} - Motifs + Coverage",
    )

    out_json = os.path.join(TMP_DIR, f"{data_id}_{gene}_chip_overlay.json")
    _write_json(out_json, json.loads(fig.to_json()))
    _progress(100)
    return {"data_id": data_id, "gene": gene, "ok": True}

def chip_overlay_compare_task(
    session_id: str,
    label: str,
    gene: str,
    window: int,
    bw_inputs: List[Tuple[str, str, str]],
    per_motif_pvals_json: str,
    min_score_bits: float,
) -> dict:
    _progress(1)
    # resolve data_id by label
    datasets = _load_session(session_id)
    try:
        data_id = next(ds.data_id for ds in datasets if ds.label == label)
    except StopIteration:
        raise RuntimeError(f"No dataset with label '{label}' in session {session_id}")

    peaks_fp = os.path.join(TMP_DIR, f"{data_id}_peaks.pkl")
    hits_fp = os.path.join(TMP_DIR, f"{data_id}_df_hits.pkl")
    _ensure(peaks_fp, "peaks"); _ensure(hits_fp, "df_hits")

    # prefer session-level motif list, fallback to per-data
    motif_list = None
    sess_motifs = os.path.join(TMP_DIR, f"{session_id}_motif_list.pkl")
    data_motifs = os.path.join(TMP_DIR, f"{data_id}_motif_list.pkl")
    if os.path.exists(sess_motifs):
        motif_list = pd.read_pickle(sess_motifs)
    elif os.path.exists(data_motifs):
        with open(data_motifs, "rb") as f:
            motif_list = pickle.load(f)
    else:
        raise RuntimeError("No cached motif_list for this session or data_id")

    peaks_df = pd.read_pickle(peaks_fp)
    df_hits = pd.read_pickle(hits_fp)

    _progress(10)
    coverage_data = fetch_bw_coverage(str(gene), peaks_df, bw_inputs)
    _progress(50)
    if not coverage_data:
        raise RuntimeError(f"No coverage data for {gene}")

    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)

    fig = plot_chip_overlay(
        coverage_data=coverage_data,
        motif_list=motif_list,
        df_hits=df_hits,
        peaks_df=peaks_df,
        peak_id=str(gene),
        window=int(window),
        per_motif_pvals=per_motif_pvals,
        min_score_bits=float(min_score_bits),
        title=f"{label}: {gene} - Motifs + Coverage",
    )

    out_json = os.path.join(TMP_DIR, f"{session_id}_{label}_{gene}_chip_overlay.json")
    _write_json(out_json, json.loads(fig.to_json()))
    _progress(100)
    return {"session_id": session_id, "label": label, "gene": gene, "ok": True}
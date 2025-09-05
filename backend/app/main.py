import os, uuid, shutil, pickle, json, csv, re, io, zipfile
import pandas as pd
from functools import reduce
from operator import and_
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request,  HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, AnyUrl
from app.new_process_input import process_genomic_input, get_motif_list
from app.new_scan_motifs import scan_wrapper
from app.filter_motifs import filter_motif_hits
from app.integrated_scoring import score_and_merge, score_hit_naive_bayes
from app.plotting import rank_peaks_for_plot, plot_occurrence_overview_plotly
from app.bigwig_overlay import fetch_bw_coverage, plot_chip_overlay
from app.run_streme import write_fasta_from_genes, run_streme_on_fasta, parse_streme_results
from app.process_tomtom import subset_by_genes
from app.filter_tfs import filter_tfs_from_gene_list
from app.utils import _parse_per_motif_pvals
from app.mem_logger import start_mem_logger

class DatasetInfo(BaseModel):
    data_id: str
    label: str
    peak_list: List[str]

class CompareInitResponse(BaseModel):
    session_id: str
    datasets: List[DatasetInfo]

class BatchScanResponse(BaseModel):
    session_id: str
    datasets: List[str]  # data_ids scanned

class ComparePlotResponse(BaseModel):
    session_id: str
    figures: Dict[str, str]           # {label: plotly_json}
    ordered_peaks: Dict[str, List[str]]  # {label: [peak_ids]}


TMP_DIR = "tmp"
TF_DATA_CACHE = {}  # type: Dict[str, pd.DataFrame]
def clear_tmp_dir():
    if os.path.isdir(TMP_DIR):
        for name in os.listdir(TMP_DIR):
            path = os.path.join(TMP_DIR, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Couldn't remove {path}: {e}")
    else:
        os.makedirs(TMP_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # your existing init work
    clear_tmp_dir()
    global TF_DATA_CACHE
    data_file = os.path.join("data", "tables2.xlsx")
    if os.path.exists(data_file):
        try:
            xls = pd.ExcelFile(data_file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                TF_DATA_CACHE[sheet_name] = df
            print(f"Loaded sheets into TF_DATA_CACHE: {list(TF_DATA_CACHE.keys())}")
        except Exception as e:
            print(f"Error loading Excel sheets: {e}")
    else:
        print("Warning: Data file not found, cache is empty.")

    # start the memory logger here
    start_mem_logger(5)

    yield  # ---- app runs ----


app = FastAPI(lifespan=lifespan)
os.makedirs(TMP_DIR, exist_ok=True)
app.mount("/tmp", StaticFiles(directory=TMP_DIR), name="tmp")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class OverviewResponse(BaseModel):
    genome: str
    peaks_df: List[Dict[str, Any]]
    data_id: str
    peak_list: List[str]

# --- Update your response model to carry Plotly JSON ---
class PlotOverviewResponse(BaseModel):
    data_id: str
    peak_list: List[str]          # ordered to match the plot
    overview_plot: str            # Plotly figure as JSON string

class ScannerResponse(BaseModel):
    data_id: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/get-genomic-input", response_model=OverviewResponse)
async def get_genomic_input(
    bed_file: Optional[UploadFile] = File(None),
    gene_list_file: Optional[UploadFile] = File(None),
    window_size: int = Form(500),
):
    # ensure our temp directory exists
    os.makedirs(TMP_DIR, exist_ok=True)

    if bed_file is None and gene_list_file is None:
        return JSONResponse(
            {"error": "Provide a gene list or a BED file"},
            status_code=400
        )
    
    # ————— handle BED upload —————
    bed_path: Optional[str] = None
    if bed_file is not None:
        bed_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.bed")
        with open(bed_path, "wb") as out:
            shutil.copyfileobj(bed_file.file, out)

    # ————— handle gene-list upload —————
    gene_list: List[str] = []
    gene_lfc: Dict[str, float] = {} #this is a dummy lfc
    if gene_list_file:
        gene_list_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.csv")
        with open(gene_list_path, "wb") as f:
            shutil.copyfileobj(gene_list_file.file, f)
        with open(gene_list_path, newline="", encoding="utf-8-sig") as inp:
            reader = csv.reader(inp, delimiter=",")
            for i, row in enumerate(reader):
                if not row or not row[0].strip():
                    continue
                first_cell = row[0].strip().lower()
                if i == 0 and ("gene" in first_cell or "symbol" in first_cell):
                    print(f"Skipping header row: {row}")
                    continue
                gene = row[0].strip()
                gene_list.append(gene)

        # Assign dummy scores: ranking for top genes
        N = len(gene_list)
        gene_lfc = {
            gene: float(N - idx)
            for idx, gene in enumerate(gene_list)
        }
        

    # ————— build peaks_df —————
    peaks_df = process_genomic_input(
        genome_filepath="fasta/genome.fa",
        gtf_filepath="fasta/dmel_genes.gtf",
        bed_path=bed_path,
        window_size=window_size,
        gene_list=gene_list,
        gene_lfc = gene_lfc
    )

    if peaks_df.empty:
        raise HTTPException(
        status_code=400,
        detail="No matching genes detected. Please check your input gene list."
    )
    # ————— persist and respond —————
    data_id = uuid.uuid4().hex
    peaks_df.to_pickle(os.path.join(TMP_DIR, f"{data_id}_peaks.pkl"))
    if gene_list:
        with open(os.path.join(TMP_DIR, f"{data_id}_genes.pkl"), "wb") as f:
            pickle.dump(gene_list, f)
        with open(os.path.join(TMP_DIR, f"{data_id}_genes_lfc.pkl"), "wb") as lf:
            pickle.dump(gene_lfc, lf) 

    def extract_gene(peak_id):
        return re.split(r"_FBtr", peak_id)[0]    
    peak_list = list(peaks_df['Peak_ID'])           
    peak_list = sorted(
        peak_list,
        key=lambda pid: abs(gene_lfc.get(extract_gene(pid), float("-inf")))
    )
    return OverviewResponse(
        genome="fasta/genome.fa",
        peaks_df=peaks_df.to_dict(orient="records"),
        data_id=data_id,
        peak_list=peak_list
    )


def _save_session(session_id: str, datasets: List[DatasetInfo]):
    with open(os.path.join(TMP_DIR, f"{session_id}_datasets.pkl"), "wb") as f:
        pickle.dump(datasets, f)

def _load_session(session_id: str) -> List[DatasetInfo]:
    try:
        with open(os.path.join(TMP_DIR, f"{session_id}_datasets.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown comparison session_id")

def _create_dataset_from_inputs(
    bed_file: Optional[UploadFile],
    gene_list_file: Optional[UploadFile],
    window_size: int
) -> tuple[str, pd.DataFrame, Dict[str, float], List[str]]:
    """
    Returns (data_id, peaks_df, gene_lfc, ordered_peak_list)
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    if bed_file is None and gene_list_file is None:
        raise HTTPException(status_code=400, detail="Provide a gene list or a BED file")

    bed_path = None
    if bed_file is not None:
        bed_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.bed")
        with open(bed_path, "wb") as out:
            shutil.copyfileobj(bed_file.file, out)

    gene_list: List[str] = []
    gene_lfc: Dict[str, float] = {}

    if gene_list_file:
        gene_list_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.csv")
        with open(gene_list_path, "wb") as f:
            shutil.copyfileobj(gene_list_file.file, f)
        with open(gene_list_path, newline="", encoding="utf-8-sig") as inp:
            reader = csv.reader(inp, delimiter=",")
            for i, row in enumerate(reader):
                if not row or not row[0].strip():
                    continue
                first_cell = row[0].strip().lower()
                if i == 0 and ("gene" in first_cell or "symbol" in first_cell):
                    continue
                gene_list.append(row[0].strip())

        # dummy rank-based LFC
        N = len(gene_list)
        gene_lfc = {g: float(N - idx) for idx, g in enumerate(gene_list)}

    peaks_df = process_genomic_input(
        genome_filepath="fasta/genome.fa",
        gtf_filepath="fasta/dmel_genes.gtf",
        bed_path=bed_path,
        window_size=window_size,
        gene_list=gene_list,
        gene_lfc=gene_lfc,
    )
    if peaks_df.empty:
        raise HTTPException(status_code=400, detail="No matching genes detected. Please check your input.")

    data_id = uuid.uuid4().hex
    peaks_df.to_pickle(os.path.join(TMP_DIR, f"{data_id}_peaks.pkl"))
    if gene_list:
        with open(os.path.join(TMP_DIR, f"{data_id}_genes.pkl"), "wb") as f:
            pickle.dump(gene_list, f)
        with open(os.path.join(TMP_DIR, f"{data_id}_genes_lfc.pkl"), "wb") as lf:
            pickle.dump(gene_lfc, lf)

    def extract_gene(pid: str) -> str:
        return re.split(r"_FBtr", pid)[0]

    peak_list = list(peaks_df["Peak_ID"])
    peak_list = sorted(peak_list, key=lambda pid: abs(gene_lfc.get(extract_gene(pid), float("-inf"))))
    return data_id, peaks_df, gene_lfc, peak_list
@app.post("/get-genomic-input-compare", response_model=CompareInitResponse)
async def get_genomic_input_compare(
    # list A
    bed_file_a: Optional[UploadFile] = File(None),
    gene_list_file_a: Optional[UploadFile] = File(None),
    label_a: str = Form("List A"),
    # list B
    bed_file_b: Optional[UploadFile] = File(None),
    gene_list_file_b: Optional[UploadFile] = File(None),
    label_b: str = Form("List B"),
    # shared
    window_size: int = Form(500),
):
    # build datasets
    data_id_a, peaks_a, lfc_a, peak_list_a = _create_dataset_from_inputs(bed_file_a, gene_list_file_a, window_size)
    data_id_b, peaks_b, lfc_b, peak_list_b = _create_dataset_from_inputs(bed_file_b, gene_list_file_b, window_size)

    session_id = uuid.uuid4().hex
    datasets = [
        DatasetInfo(data_id=data_id_a, label=label_a, peak_list=peak_list_a),
        DatasetInfo(data_id=data_id_b, label=label_b, peak_list=peak_list_b),
    ]
    _save_session(session_id, datasets)
    return CompareInitResponse(session_id=session_id, datasets=datasets)

@app.post("/validate-motifs-group")
async def validate_motifs_group(
    motifs: list[str] = Form(...),
    session_id: str = Form(...),
):
    try:
        motif_inputs = [json.loads(m) for m in motifs]
        motif_list = get_motif_list(motif_inputs)
        with open(f"{TMP_DIR}/{session_id}_motif_list.pkl", "wb") as f:
            pickle.dump(motif_list, f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid motifs: {e}")
    return {"status": "ok", "count": len(motif_list)}


@app.post("/validate-motifs")
async def validate_motifs(
    motifs: list[str] = Form(...),
    data_id: str = Form(...)
):
    # parse JSON strings into dicts
    try:
        motif_inputs = [json.loads(m) for m in motifs]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed motif JSON: {e}")
    try:
        motif_list = get_motif_list(motif_inputs)
        with open(f"{TMP_DIR}/{data_id}_motif_list.pkl", "wb") as f:
            pickle.dump(motif_list, f)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "count": len(motif_list)}

@app.post("/get-motif-hits-batch", response_model=BatchScanResponse)
async def get_motif_hits_batch(
    session_id: str = Form(...),
    window: int = Form(500),
    fimo_threshold: float = Form(0.005),
):
    datasets = _load_session(session_id)

    # load session-level motifs
    try:
        motif_list = pd.read_pickle(f"{TMP_DIR}/{session_id}_motif_list.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Run /validate-motifs-group first for this session")

    scanned_ids = []
    for ds in datasets:
        try:
            peaks_df = pd.read_pickle(f"{TMP_DIR}/{ds.data_id}_peaks.pkl")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"No cached peaks for {ds.data_id}")

        motif_hits, df_hits = scan_wrapper(peaks_df, "fasta/genome.fa", window, motif_list, fimo_threshold)

        with open(f"{TMP_DIR}/{ds.data_id}_motif_hits.pkl", "wb") as f:
            pickle.dump(motif_hits, f)
        with open(f"{TMP_DIR}/{ds.data_id}_df_hits.pkl", "wb") as f:
            pickle.dump(df_hits, f)
        scanned_ids.append(ds.data_id)

    return BatchScanResponse(session_id=session_id, datasets=scanned_ids)

@app.post("/get-motif-hits", response_model=ScannerResponse)
async def get_motif_hits(
    motifs: list[str] = Form(...),
    window: int = Form(500),
    fimo_threshold: float = Form(0.005),
    data_id: str = Form(...)
    ):
    if not motifs:
        raise HTTPException(status_code=400, detail="At least one motif is required")

    try:
        motif_list = pd.read_pickle(f"{TMP_DIR}/{data_id}_motif_list.pkl")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid motifs: {e}")

    try:
        peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached peak data here - try after running Process Genomic Input!")

    motif_hits, df_hits = scan_wrapper(peaks_df, "fasta/genome.fa", window, motif_list, fimo_threshold)


    with open(f"{TMP_DIR}/{data_id}_motif_hits.pkl", "wb") as f:
        pickle.dump(motif_hits, f)
    with open(f"{TMP_DIR}/{data_id}_df_hits.pkl", "wb") as f:
        pickle.dump(df_hits, f)
    return {
        "data_id": data_id
    }



@app.post("/plot-motif-overview-compare", response_model=ComparePlotResponse)
async def plot_motif_overview_compare(
    session_id: str = Form(...),
    window: int = Form(500),
    per_motif_pvals_json: str = Form(default="")
):
    datasets = _load_session(session_id)

    # shared motif list
    try:
        motif_list = pd.read_pickle(f"{TMP_DIR}/{session_id}_motif_list.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached motif_list for this session")

    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)

    figures: Dict[str, str] = {}
    ordered: Dict[str, List[str]] = {}

    for ds in datasets:
        data_id = ds.data_id
        label = ds.label

        try:
            peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
            df_hits  = pd.read_pickle(f"{TMP_DIR}/{data_id}_df_hits.pkl")
            gene_lfc = pd.read_pickle(f"{TMP_DIR}/{data_id}_genes_lfc.pkl")
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"Cache miss for dataset {label}: {e}")

        peak_ranks = rank_peaks_for_plot(
            df_hits=df_hits,
            gene_lfc=gene_lfc,
            peaks_df=peaks_df,
            use_hit_number=False,
            use_match_score=False,
            motif=None,
            min_score_bits=0.0,
            per_motif_pvals=per_motif_pvals,
        )

        fig, ordered_peaks = plot_occurrence_overview_plotly(
            peak_list=list(peak_ranks.keys()),
            peaks_df=peaks_df,
            motif_list=motif_list,
            df_hits=df_hits,
            window=window,
            peak_rank=peak_ranks,
            title=f"Motif hits — {label}",
            min_score_bits=0.0,
            per_motif_pvals=per_motif_pvals,
        )

        figures[label] = fig.to_json()
        ordered[label] = ordered_peaks

    return ComparePlotResponse(session_id=session_id, figures=figures, ordered_peaks=ordered)

@app.post("/plot-motif-overview", response_model=PlotOverviewResponse)
async def plot_motif_overview(
    request: Request,
    window: int = Form(500),
    data_id: str = Form(...),
    per_motif_pvals_json: str = Form(default="")   # NEW
):
    # Load cached data
    try:
        peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached peaks for given data_id")

    try:
        df_hits = pd.read_pickle(f"{TMP_DIR}/{data_id}_df_hits.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached motif_hits for given data_id")

    try:
        motif_list = pd.read_pickle(f"{TMP_DIR}/{data_id}_motif_list.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached motif_list for given data_id")

    try:
        gene_lfc = pd.read_pickle(f"{TMP_DIR}/{data_id}_genes_lfc.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached gene_logfc dict for given data_id")

    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)
    peak_ranks = rank_peaks_for_plot(
        df_hits=df_hits,
        gene_lfc=gene_lfc,
        peaks_df=peaks_df,
        use_hit_number=False,
        use_match_score=False,
        motif=None,
        min_score_bits=0.0,
        per_motif_pvals=per_motif_pvals,    # NEW
    )

    fig, ordered_peaks = plot_occurrence_overview_plotly(
        peak_list=list(peak_ranks.keys()),
        peaks_df=peaks_df,
        motif_list=motif_list,
        df_hits=df_hits,
        window=window,
        peak_rank=peak_ranks,
        title="Motif hits",
        min_score_bits=0.0,
        per_motif_pvals=per_motif_pvals,    # NEW
    )

    # Serialize Plotly figure to JSON
    fig_json = fig.to_json()       # string; frontend will parse and render

    return {
        "overview_plot": fig_json,
        "data_id": data_id,
        "peak_list": ordered_peaks
    }

@app.post("/filter-motif-hits", response_model=ScannerResponse)
async def filter_and_score(
    atac_bed: UploadFile = File(None),      # now optional
    chip_bed: UploadFile = File(None),      # now optional
    data_id: str = Form(...),
):
    if not atac_bed and not chip_bed:
        raise HTTPException(
            status_code=400,
            detail="You must provide at least one of ATAC or ChIP BED files."
        )
    try:
        with open(f"{TMP_DIR}/{data_id}_df_hits.pkl","rb") as f:
            df_hits = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Missing cache for data_id")

    empty_hits = pd.DataFrame(columns=df_hits.columns)

    if atac_bed:
        atac_path = os.path.join(TMP_DIR, f"{data_id}_atac.bed")
        with open(atac_path,"wb") as f:
            f.write(await atac_bed.read())
        atac_filt_hits = filter_motif_hits(df_hits, atac_path)
        with open(f"{TMP_DIR}/{data_id}_atac_filt_hits.pkl","wb") as f:
            pickle.dump(atac_filt_hits, f)
    else:
        atac_filt_hits = empty_hits

    if chip_bed:
        chip_path = os.path.join(TMP_DIR, f"{data_id}_chip.bed")
        with open(chip_path,"wb") as f:
            f.write(await chip_bed.read())
        chip_filt_hits = filter_motif_hits(df_hits, chip_path)
        with open(f"{TMP_DIR}/{data_id}_chip_filt_hits.pkl","wb") as f:
            pickle.dump(chip_filt_hits, f)
    else:
        chip_filt_hits = empty_hits

    scored_df = score_and_merge(df_hits, chip_filt_hits, atac_filt_hits)

    ranked_df = score_hit_naive_bayes(scored_df)
    top_hits = (
        ranked_df
        .reset_index()
        .sort_values("P_regulatory", ascending=False)
        [["Peak_ID", "Motif", "P_regulatory", "M_prom", "M_chip", "M_atac", "logFC", "FIMO_score"]]
    )
    top_hits.to_csv(f"{TMP_DIR}/{data_id}_top_hits.tsv", sep="\t", index=False)

    return {"data_id": data_id}

@app.post("/filter-motif-hits-batch", response_model=BatchScanResponse)
async def filter_and_score_batch(
    session_id: str = Form(...),
    atac_bed_shared: UploadFile = File(None),
    chip_bed_shared: UploadFile = File(None),
    atac_bed_a: UploadFile = File(None),
    chip_bed_a: UploadFile = File(None),
    atac_bed_b: UploadFile = File(None),
    chip_bed_b: UploadFile = File(None),
):
    datasets = _load_session(session_id)
    if not datasets:
        raise HTTPException(status_code=404, detail="Empty/unknown session")

    any_shared = bool(atac_bed_shared or chip_bed_shared)
    any_ab     = bool(atac_bed_a or chip_bed_a or atac_bed_b or chip_bed_b)
    if not (any_shared or any_ab):
        raise HTTPException(status_code=400, detail="Provide at least one ATAC or ChIP BED.")

    # ---------- helper ----------
    async def _filter_one_dataset(data_id: str, atac: bytes | None, chip: bytes | None) -> None:
        try:
            with open(f"{TMP_DIR}/{data_id}_df_hits.pkl","rb") as f:
                df_hits = pickle.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Missing cache for data_id {data_id}")

        empty_hits = pd.DataFrame(columns=df_hits.columns)

        # ATAC
        if atac:
            atac_path = os.path.join(TMP_DIR, f"{data_id}_atac.bed")
            with open(atac_path, "wb") as f:
                f.write(atac)
            atac_filt_hits = filter_motif_hits(df_hits, atac_path)
        else:
            atac_filt_hits = empty_hits
        with open(f"{TMP_DIR}/{data_id}_atac_filt_hits.pkl","wb") as f:
            pickle.dump(atac_filt_hits, f)

        # ChIP
        if chip:
            chip_path = os.path.join(TMP_DIR, f"{data_id}_chip.bed")
            with open(chip_path, "wb") as f:
                f.write(chip)
            chip_filt_hits = filter_motif_hits(df_hits, chip_path)
        else:
            chip_filt_hits = empty_hits
        with open(f"{TMP_DIR}/{data_id}_chip_filt_hits.pkl","wb") as f:
            pickle.dump(chip_filt_hits, f)

        # scoring
        scored_df = score_and_merge(df_hits, chip_filt_hits, atac_filt_hits)
        ranked_df = score_hit_naive_bayes(scored_df)
        top_hits = (
            ranked_df.reset_index()
                     .sort_values("P_regulatory", ascending=False)
                     [["Peak_ID","Motif","P_regulatory","M_prom","M_chip","M_atac","logFC","FIMO_score"]]
        )
        top_hits.to_csv(f"{TMP_DIR}/{data_id}_top_hits.tsv", sep="\t", index=False)

    processed: list[str] = []

    if any_shared:
        # Read ONCE, reuse the bytes across all datasets
        atac_bytes = await atac_bed_shared.read() if atac_bed_shared else None
        chip_bytes = await chip_bed_shared.read() if chip_bed_shared else None

        for ds in datasets:
            await _filter_one_dataset(ds.data_id, atac_bytes, chip_bytes)
            processed.append(ds.data_id)
    else:
        if len(datasets) != 2:
            raise HTTPException(status_code=400, detail="Per-dataset A/B uploads require exactly two datasets.")
        # Read each upload ONCE
        a_atac = await atac_bed_a.read() if atac_bed_a else None
        a_chip = await chip_bed_a.read() if chip_bed_a else None
        b_atac = await atac_bed_b.read() if atac_bed_b else None
        b_chip = await chip_bed_b.read() if chip_bed_b else None

        await _filter_one_dataset(datasets[0].data_id, a_atac, a_chip)
        await _filter_one_dataset(datasets[1].data_id, b_atac, b_chip)
        processed.extend([datasets[0].data_id, datasets[1].data_id])

    return BatchScanResponse(session_id=session_id, datasets=processed)

@app.get("/download-top-hits/{data_id}")
async def download_top_hits(data_id: str, label: str | None = None):
    file_path = os.path.join(TMP_DIR, f"{data_id}_top_hits.tsv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Top hits file not found")

    safe_label = label or data_id  # fallback if label not provided
    filename = f"{safe_label}_top_hits.tsv"

    return FileResponse(
        file_path,
        media_type="text/tab-separated-values",
        filename=filename,
    )

# --- Batch zip of all top_hits in the session ---
@app.get("/download-top-hits-batch")
def download_top_hits_batch(session_id: str):
    datasets = _load_session(session_id)
    if not datasets:
        raise HTTPException(status_code=404, detail="Unknown or empty session_id")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        added_any = False
        for ds in datasets:
            data_id = ds.data_id
            p = os.path.join(TMP_DIR, f"{data_id}_top_hits.tsv")
            if os.path.exists(p):
                # Name each file clearly inside the zip; fall back to data_id
                inner_name = f"{getattr(ds, 'label', data_id)}_top_hits.tsv"
                zf.write(p, arcname=inner_name)
                added_any = True

    if not added_any:
        raise HTTPException(status_code=404, detail="No top_hits files found for this session.")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="top_hits_{session_id}.zip"'
        },
    )

def load_filtered_hits(data_id: str, modality: str) -> pd.DataFrame:
    # normalize kind
    kind = modality.lower()
    if kind not in {"atac","chip"}:
        raise ValueError(f"Unknown kind {kind}")

    p = os.path.join(TMP_DIR, f"{data_id}_{modality}_filt_hits.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing filtered hits: {p}")

    df = pd.read_pickle(p)
    # sanity: ensure expected cols exist
    required = {"Peak_ID","Motif"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns: {missing}")

    return df

@app.post("/plot-filtered-overview-compare", response_model=ComparePlotResponse)
async def plot_filtered_overview_compare(
    session_id: str = Form(...),
    window: int = Form(500),
    chip: str = Form("false"),
    atac: str = Form("false"),
    use_hit_number: str = Form("false"),
    use_match_score: str = Form("false"),
    chosen_motif: str = Form(""),
    per_motif_pvals_json: str = Form(default=""),
):
    chip            = chip.lower() == "true"
    atac            = atac.lower() == "true"
    use_hit_number  = use_hit_number.lower() == "true"
    use_match_score = use_match_score.lower() == "true"

    datasets = _load_session(session_id)

    try:
        motif_list = pd.read_pickle(f"{TMP_DIR}/{session_id}_motif_list.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached motif_list for this session")

    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)

    figures: Dict[str, str] = {}
    ordered: Dict[str, List[str]] = {}

    for ds in datasets:
        data_id = ds.data_id
        label   = ds.label

        try:
            peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
            gene_lfc = pd.read_pickle(f"{TMP_DIR}/{data_id}_genes_lfc.pkl")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Missing cached genomic data for {label}")

        # choose which hits table to plot
        if not (chip or atac):
            try:
                df_to_plot = pd.read_pickle(f"{TMP_DIR}/{data_id}_df_hits.pkl")
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"No cached df_hits for {label}")
        else:
            dfs = []
            if chip: dfs.append(load_filtered_hits(data_id, "chip"))
            if atac: dfs.append(load_filtered_hits(data_id, "atac"))
            df_to_plot = reduce(lambda L, R: pd.merge(L, R, how="inner"), dfs) if len(dfs) > 1 else dfs[0]

        peak_ranks = rank_peaks_for_plot(
            df_hits=df_to_plot,
            gene_lfc=gene_lfc,
            peaks_df=peaks_df,
            use_hit_number=use_hit_number,
            use_match_score=use_match_score,
            motif=(chosen_motif or None),
            min_score_bits=0.0,
            per_motif_pvals=per_motif_pvals,
        )

        fig, ordered_peaks = plot_occurrence_overview_plotly(
            peak_list=list(peak_ranks.keys()),
            peaks_df=peaks_df,
            motif_list=motif_list,
            df_hits=df_to_plot,
            window=window,
            peak_rank=peak_ranks,
            title=f"Motif hits — {label}",
            min_score_bits=0.0,
            per_motif_pvals=per_motif_pvals,
        )

        figures[label] = fig.to_json()
        ordered[label] = ordered_peaks

    return ComparePlotResponse(session_id=session_id, figures=figures, ordered_peaks=ordered)


@app.post("/plot-filtered-overview", response_model=PlotOverviewResponse)
async def plot_filtered_overview(
    request: Request,
    chip: str = Form("false"),
    atac: str = Form("false"),
    window: int = Form(500),
    use_hit_number: str = Form("false"),
    use_match_score: str = Form("false"),
    chosen_motif: str = Form(...),
    data_id: str = Form(...),
    per_motif_pvals_json: str = Form(default="")   # NEW
):
    chip             = chip.lower() == "true"
    atac             = atac.lower() == "true"
    use_hit_number   = use_hit_number.lower() == "true"
    use_match_score  = use_match_score.lower() == "true"

    try:
        peaks_df   = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_peaks.pkl"))
        gene_lfc   = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_genes_lfc.pkl"))
        motif_list = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_motif_list.pkl"))
    except FileNotFoundError:
        raise HTTPException(404, "No cached genomic data for given data_id")

    if not (chip or atac):
        df_to_plot = pd.read_pickle(f"{TMP_DIR}/{data_id}_df_hits.pkl")

    else:
        dfs = []
        if chip: dfs.append(load_filtered_hits(data_id, "chip"))
        if atac: dfs.append(load_filtered_hits(data_id, "atac"))
        df_to_plot = reduce(lambda L, R: pd.merge(L, R, how="inner"), dfs) if len(dfs) > 1 else dfs[0]

    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)

    peak_ranks = rank_peaks_for_plot(
        df_hits=df_to_plot,
        gene_lfc=gene_lfc,
        peaks_df=peaks_df,
        use_hit_number=use_hit_number,
        use_match_score=use_match_score,
        motif=chosen_motif,
        min_score_bits=0.0,
        per_motif_pvals=per_motif_pvals,    
    )

    fig, ordered_peaks = plot_occurrence_overview_plotly(
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

    fig_json = fig.to_json()
    return {"overview_plot": fig_json, "data_id": data_id, "peak_list": ordered_peaks}

@app.post("/plot-chip-overlay")
async def plot_chip_overlay_json(
    request: Request,
    data_id: str = Form(...),
    bigwigs: List[UploadFile] = File(...),   # uploaded .bw files
    chip_tracks: List[str] = Form(...),      # each "Label|#RRGGBB"
    gene: str = Form(...),                   # Peak_ID
    window: int = Form(500),
    # optional filters to match your new plotting.py
    per_motif_pvals_json: str = Form(default=""),
    min_score_bits: float = Form(0.0),
):
    # ---- load cached inputs (new pipeline) ----
    try:
        peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached peaks for given data_id")

    try:
        with open(f"{TMP_DIR}/{data_id}_motif_list.pkl", "rb") as f:
            motif_list = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached motif_list for given data_id")

    try:
        df_hits = pd.read_pickle(f"{TMP_DIR}/{data_id}_df_hits.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached motif_hits (df_hits) for given data_id")

    # ---- ingest uploaded bigWigs ----
    if len(bigwigs) != len(chip_tracks):
        raise HTTPException(status_code=400, detail="Mismatch in bigwig and track metadata count")

    os.makedirs(TMP_DIR, exist_ok=True)
    bw_input: List[Tuple[str, str, str]] = []
    try:
        for i, bw_file in enumerate(bigwigs):
            try:
                label, color = chip_tracks[i].split("|", 1)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid track meta '{chip_tracks[i]}'. Use 'Label|#RRGGBB'.")

            bw_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.bw")
            with open(bw_path, "wb") as f:
                shutil.copyfileobj(bw_file.file, f)
            bw_input.append((bw_path, label.strip(), color.strip()))
    finally:
        # close upload handles
        for uf in bigwigs:
            try:
                uf.file.close()
            except Exception:
                pass

    # ---- build coverage for selected peak ----
    coverage_data = fetch_bw_coverage(str(gene), peaks_df, bw_input)
    # clean temp bigwigs (coverage_data holds arrays, files no longer needed)
    for path, _, _ in bw_input:
        try:
            os.remove(path)
        except Exception:
            pass

    if not coverage_data:
        raise HTTPException(status_code=400, detail=f"No coverage data for {gene}")

    # ---- filters ----
    per_motif_pvals = _parse_per_motif_pvals(per_motif_pvals_json)

    # ---- make plotly fig ----
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


    fig_json = fig.to_json()

    return {
        "chip_overlay_plot": fig_json,   # frontend parses and renders
        "data_id": data_id,
        "gene": gene,
        "tracks": [t.split("|", 1)[0] for t in chip_tracks],
        "window": int(window),
        "applied_pvals": per_motif_pvals or {},
        "min_score_bits": float(min_score_bits),
    }


@app.post("/plot-chip-overlay-compare")
async def plot_chip_overlay_compare_json(
    request: Request,
    session_id: str = Form(...),
    label: str = Form(...),                 # "List A" or "List B" (or any label you used)
    bigwigs: List[UploadFile] = File(...),  # uploaded .bw files
    chip_tracks: List[str] = Form(...),     # each "Label|#RRGGBB"
    gene: str = Form(...),                  # Peak_ID (from the chosen label's dataset)
    window: int = Form(500),
    per_motif_pvals_json: str = Form(default=""),
    min_score_bits: float = Form(0.0),
):
    # --- resolve data_id from session+label ---
    datasets = _load_session(session_id)
    try:
        data_id = next(ds.data_id for ds in datasets if ds.label == label)
    except StopIteration:
        raise HTTPException(status_code=404, detail=f"No dataset with label '{label}' in session {session_id}")

    # ---- load cached inputs (same as single-mode endpoint) ----
    try:
        peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached peaks for given data_id")

    # motif list can be session-scoped (group-validated) or per-data_id; prefer session first
    motif_list = None
    try:
        motif_list = pd.read_pickle(f"{TMP_DIR}/{session_id}_motif_list.pkl")
    except FileNotFoundError:
        # fallback: per-data_id
        try:
            with open(f"{TMP_DIR}/{data_id}_motif_list.pkl", "rb") as f:
                motif_list = pickle.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No cached motif_list for this session or data_id")

    try:
        df_hits = pd.read_pickle(f"{TMP_DIR}/{data_id}_df_hits.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached motif_hits (df_hits) for given data_id")

    # ---- ingest uploaded bigWigs ----
    if len(bigwigs) != len(chip_tracks):
        raise HTTPException(status_code=400, detail="Mismatch in bigwig and track metadata count")

    os.makedirs(TMP_DIR, exist_ok=True)
    bw_input: List[Tuple[str, str, str]] = []
    try:
        for i, bw_file in enumerate(bigwigs):
            try:
                tr_label, color = chip_tracks[i].split("|", 1)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid track meta '{chip_tracks[i]}'. Use 'Label|#RRGGBB'.")
            bw_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.bw")
            with open(bw_path, "wb") as f:
                shutil.copyfileobj(bw_file.file, f)
            bw_input.append((bw_path, tr_label.strip(), color.strip()))
    finally:
        for uf in bigwigs:
            try: uf.file.close()
            except Exception: pass

    # ---- build coverage for selected peak ----
    coverage_data = fetch_bw_coverage(str(gene), peaks_df, bw_input)
    for path, _, _ in bw_input:
        try: os.remove(path)
        except Exception: pass

    if not coverage_data:
        raise HTTPException(status_code=400, detail=f"No coverage data for {gene}")

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

    return {
        "chip_overlay_plot": fig.to_json(),
        "session_id": session_id,
        "label": label,
        "data_id": data_id,
        "gene": gene,
        "tracks": [t.split("|", 1)[0] for t in chip_tracks],
        "window": int(window),
        "applied_pvals": per_motif_pvals or {},
        "min_score_bits": float(min_score_bits),
    }

@app.get("/get-tissues")
async def get_tissues(
    data_source: str = Query(...),
    stage: str = Query(...)
):
    sheet_map = {
        "10-12": "stage 10-12",
        "13-16": "stage 13-16"
    }

    if stage not in sheet_map:
        raise HTTPException(status_code=400, detail="Invalid stage")

    sheet_name = sheet_map[stage]
    if sheet_name not in TF_DATA_CACHE:
        raise HTTPException(status_code=404, detail="Stage data not loaded")

    df = TF_DATA_CACHE[sheet_name]

    if "cell_types" not in df.columns:
        raise HTTPException(status_code=500, detail="Missing 'cell_types' column")

    tissues = sorted(df["cell_types"].dropna().unique().tolist())
    return {"tissues": tissues}


@app.get("/get-de-genes")
async def get_de_genes(
    data_source: str = Query(...),
    stage: str = Query(...),
    tissue: str = Query(...)
):
    sheet_map = {
        "10-12": "stage 10-12",
        "13-16": "stage 13-16"
    }

    if stage not in sheet_map:
        raise HTTPException(status_code=400, detail="Invalid stage")

    sheet_name = sheet_map[stage]
    if sheet_name not in TF_DATA_CACHE:
        raise HTTPException(status_code=404, detail="Stage data not loaded")

    df = TF_DATA_CACHE[sheet_name]

    if "cell_types" not in df.columns or "gene" not in df.columns or "avg_log2FC" not in df.columns:
        raise HTTPException(status_code=500, detail="Missing required columns in data")

    filtered_df = df[df["cell_types"] == tissue][["gene", "avg_log2FC"]]

    genes = (
        filtered_df.to_dict(orient="records")
        if not filtered_df.empty
        else []
    )

    return {"genes": genes}

@app.get("/download-de-genes")
async def download_de_genes(
    data_source: str = Query(...),
    stage: str = Query(...),
    tissue: str = Query(...)
):
    sheet_map = {
        "10-12": "stage 10-12",
        "13-16": "stage 13-16"
    }

    if stage not in sheet_map:
        raise HTTPException(status_code=400, detail="Invalid stage")

    sheet_name = sheet_map[stage]
    if sheet_name not in TF_DATA_CACHE:
        raise HTTPException(status_code=404, detail="Stage data not loaded")

    df = TF_DATA_CACHE[sheet_name]

    if "cell_types" not in df.columns or "gene" not in df.columns or "avg_log2FC" not in df.columns:
        raise HTTPException(status_code=500, detail="Missing required columns in data")

    filtered_df = df[df["cell_types"] == tissue][["gene", "avg_log2FC"]]

    if filtered_df.empty:
        csv_content = "gene,avg_log2FC\n"
    else:
        csv_content = filtered_df.to_csv(index=False)

    filename = f"{tissue}_{stage}_DE_genes.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/run-streme")
async def run_streme_endpoint(
    request: Request,
    minw: int = Form(6),
    maxw: int = Form(10),
    window_size: int = Form(500),
    use_de_genes: bool = Form(True),
    tissue: Optional[str] = Form(None),
    stage: Optional[str] = Form(None),
    gene_file: Optional[UploadFile] = File(None)
):
    try:
        genome_path = "fasta/genome.fa"

        if use_de_genes:
            if not tissue or not stage:
                raise HTTPException(status_code=400, detail="Missing tissue or stage selection")

            # Retrieve DE genes from cache (assuming TF_DATA_CACHE uses sheet names)
            sheet_map = {
                "10-12": "stage 10-12",
                "13-16": "stage 13-16"
            }

            if stage not in sheet_map:
                raise HTTPException(status_code=400, detail="Invalid stage")

            sheet_name = sheet_map[stage]
            if sheet_name not in TF_DATA_CACHE:
                raise HTTPException(status_code=404, detail="Stage data not loaded")

            df = TF_DATA_CACHE[sheet_name]
            if "gene" not in df.columns or "cell_types" not in df.columns:
                raise HTTPException(status_code=500, detail="Required columns missing in DE data")

            gene_list = (df[df["cell_types"] == tissue]["gene"].dropna().head(250).tolist())

            if not gene_list:
                raise HTTPException(status_code=404, detail="No DE genes found for this tissue")

        else:
            if gene_file is None:
                raise HTTPException(status_code=400, detail="Gene list file missing")

            df = pd.read_csv(gene_file.file)
            if df.empty or df.shape[1] < 1:
                raise HTTPException(status_code=400, detail="Uploaded gene file is empty or invalid")

            gene_list = df.iloc[:, 0].dropna().tolist()

        # Write FASTA from genes
        fasta_path = write_fasta_from_genes(gene_list, genome_path, window_size)

        # Run STREME
        streme_out, output_id = run_streme_on_fasta(fasta_path, minw, maxw, TMP_DIR)

        # Parse STREME results
        motifs, html_url = parse_streme_results(streme_out, output_id, request)

        return {"motifs": motifs, "streme_html_url": html_url, "tmp_id": output_id}

    except HTTPException as e:
        raise e

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/download-streme/{tmp_id}")
def download_streme_file(tmp_id: str):
    file_path = os.path.join(TMP_DIR, f"{tmp_id}_streme_out", "streme.txt")
    if not os.path.isfile(file_path):
        return Response(content=f"File not found: {file_path}", status_code=404)
    return FileResponse(path=file_path, filename=f"streme_{tmp_id}.txt", media_type="text/plain")

@app.post("/process-tomtom")
async def process_tomtom(
    tsv_file: UploadFile = File(...),
    gene_list_json: str   = Form(...),
):
    gene_list = json.loads(gene_list_json)
    print(gene_list)
    # ← read into a DataFrame from the UploadFile
    df = pd.read_csv(tsv_file.file, sep="\t", dtype=str)
    df = df.dropna(subset=["Target_ID"])
    # ← call the new helper
    sub, motif_gene_pairs = subset_by_genes(
        df, "fasta/dmel_genes.gtf", gene_list
    )
    print(sub)
    out = (
        motif_gene_pairs
        .rename(columns={
            "Query_ID": "motif",
            "gene_symbol": "gene"
        })
        .to_dict(orient="records")
    )

    return {"pairs": out}

@app.post("/filter-tfs")
async def filter_tfs(
    use_de_genes: bool = Form(...),
    tissue: str = Form(None),
    stage: str = Form(None),
    gene_file: UploadFile = None
):
    gene_list = []
    if use_de_genes:
        if not tissue or not stage:
            raise HTTPException(status_code=400, detail="Missing tissue or stage selection")

        # Retrieve DE genes from cache (assuming TF_DATA_CACHE uses sheet names)
        sheet_map = {
            "10-12": "stage 10-12",
            "13-16": "stage 13-16"
        }

        if stage not in sheet_map:
            raise HTTPException(status_code=400, detail="Invalid stage")

        sheet_name = sheet_map[stage]
        if sheet_name not in TF_DATA_CACHE:
            raise HTTPException(status_code=404, detail="Stage data not loaded")

        df = TF_DATA_CACHE[sheet_name]
        if "gene" not in df.columns or "cell_types" not in df.columns:
            raise HTTPException(status_code=500, detail="Required columns missing in DE data")

        gene_list = (df[df["cell_types"] == tissue]["gene"].dropna().head(250).tolist())

        if not gene_list:
            raise HTTPException(status_code=404, detail="No DE genes found for this tissue")

    else:
        if gene_file is None:
            raise HTTPException(status_code=400, detail="Gene list file missing")

        df = pd.read_csv(gene_file.file)
        if df.empty or df.shape[1] < 1:
            raise HTTPException(status_code=400, detail="Uploaded gene file is empty or invalid")

        gene_list = df.iloc[:, 0].dropna().tolist()

    tf_results = filter_tfs_from_gene_list(gene_list)
    print(tf_results)
    return {"tfs": tf_results}


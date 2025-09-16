import os, uuid, shutil, pickle, json, csv, re, io, zipfile
import pandas as pd
from functools import reduce
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request,  HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
from redis import Redis
from rq import Queue, Retry
from rq.job import Job
from app.tasks import (
    scan_session_batch, 
    scan_single_dataset,
    plot_overview_single_task,
    plot_overview_compare_task,
    filter_score_single_task, 
    filter_score_batch_task,
    plot_filtered_single_task, 
    plot_filtered_compare_task)
from app.new_process_input import process_genomic_input, get_motif_list
from app.new_scan_motifs import scan_wrapper
from app.filter_motifs import filter_motif_hits
from app.integrated_scoring import score_and_merge, score_hit_naive_bayes
from app.plotting import rank_peaks_for_plot, plot_occurrence_overview_plotly
from app.bigwig_overlay import fetch_bw_coverage, plot_chip_overlay
from app.run_streme import write_fasta_from_genes, run_streme_on_fasta, parse_streme_results
from app.process_tomtom import subset_by_genes
from app.filter_tfs import filter_tfs_from_gene_list
from app.utils import clear_tmp_dir, _parse_per_motif_pvals, _df_to_records_json_safe, _df_to_csv_bytes
from app.mem_logger import start_mem_logger
import app.models as models

TMP_DIR = "tmp"
TF_DATA_CACHE = {}  # type: Dict[str, pd.DataFrame]

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    start_mem_logger(5)
    yield 

# create a Redis connection and an RQ queue (do once)
redis_conn = Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", "6379")), db=0)
q = Queue("cpu", connection=redis_conn, default_timeout="3600")  # 1 hour default timeout


app = FastAPI(lifespan=lifespan)
os.makedirs(TMP_DIR, exist_ok=True)
#app.mount("/tmp", StaticFiles(directory=TMP_DIR), name="tmp")
#app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.mount("/api/tmp", StaticFiles(directory=TMP_DIR), name="tmp")
app.mount("/api/static", StaticFiles(directory="app/static"), name="static")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/health")
def health_api():
    return {"ok": True, "path": "/api/health"}

@app.post("/get-genomic-input", response_model=models.OverviewResponse)
async def get_genomic_input(
    bed_file: Optional[UploadFile] = File(None),
    gene_list_file: Optional[UploadFile] = File(None),
    window_size: int = Form(500),
):
    os.makedirs(TMP_DIR, exist_ok=True)

    if bed_file is None and gene_list_file is None:
        return JSONResponse(
            {"error": "Provide a gene list or a BED file"},
            status_code=400
        )
    # handle bed upload
    bed_path: Optional[str] = None
    if bed_file is not None:
        bed_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.bed")
        with open(bed_path, "wb") as out:
            shutil.copyfileobj(bed_file.file, out)

    # handle gene-list upload 
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

        # assign dummy scores: ranking for top genes
        N = len(gene_list)
        gene_lfc = {
            gene: float(N - idx)
            for idx, gene in enumerate(gene_list)
        }      

    # build peaks_df
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
    return models.OverviewResponse(
        genome="fasta/genome.fa",
        peaks_df=peaks_df.to_dict(orient="records"),
        data_id=data_id,
        peak_list=peak_list
    )


def _save_session(session_id: str, datasets: List[models.DatasetInfo]):
    with open(os.path.join(TMP_DIR, f"{session_id}_datasets.pkl"), "wb") as f:
        pickle.dump(datasets, f)

def _load_session(session_id: str) -> List[models.DatasetInfo]:
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

@app.post("/get-genomic-input-compare", response_model=models.CompareInitResponse)
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
        models.DatasetInfo(data_id=data_id_a, label=label_a, peak_list=peak_list_a),
        models.DatasetInfo(data_id=data_id_b, label=label_b, peak_list=peak_list_b),
    ]
    _save_session(session_id, datasets)
    return models.CompareInitResponse(session_id=session_id, datasets=datasets)

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


@app.post("/get-motif-hits", response_model=models.EnqueueResponse)
async def get_motif_hits(
    motifs: list[str] = Form(...),
    window: int = Form(500),
    fimo_threshold: float = Form(0.005),
    data_id: str = Form(...)
):
    if not motifs:
        raise HTTPException(status_code=400, detail="At least one motif is required")

    # (light) preflight checks to fail fast
    try:
        pd.read_pickle(f"{TMP_DIR}/{data_id}_motif_list.pkl")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid motifs for {data_id}: {e}")

    try:
        pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached peak data here - run Process Genomic Input!")

    # enqueue the heavy job
    job = q.enqueue(
        scan_single_dataset,
        data_id,
        window,
        fimo_threshold,
        retry=Retry(max=3, interval=[10, 30, 60]),
        job_timeout="600"  # 10 min; adjust acc to workload
    )
    return {"job_id": job.get_id()}

@app.post("/get-motif-hits-batch", response_model=models.EnqueueResponse)
async def get_motif_hits_batch(
    session_id: str = Form(...),
    window: int = Form(500),
    fimo_threshold: float = Form(0.005),
):
    # load datasets list first 
    datasets = _load_session(session_id)
    dataset_ids = [ds.data_id for ds in datasets]

    # preflight checks
    try:
        pd.read_pickle(f"{TMP_DIR}/{session_id}_motif_list.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Run /validate-motifs-group first for this session")

    for dsid in dataset_ids:
        if not os.path.exists(f"{TMP_DIR}/{dsid}_peaks.pkl"):
            raise HTTPException(status_code=404, detail=f"No cached peaks for {dsid}")

    # enqueue ONE job to process all datasets for this session
    job = q.enqueue(
        scan_session_batch,
        session_id,
        dataset_ids,
        window,
        fimo_threshold,
        retry=Retry(max=3, interval=[10, 30, 60]),
        job_timeout="1200"  # 20 min for big batches
    )
    return {"job_id": job.get_id()}

# a small status endpoint to poll job state
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    state = job.get_status()  # 'queued' | 'started' | 'finished' | 'failed' | 'deferred'
    resp = {
        "job_id": job_id,
        "status": state,
        "progress": job.meta.get("progress", None),
    }
    if state == "failed":
        resp["error"] = str(job.exc_info)[-2000:] if job.exc_info else "Unknown error"
    if state == "finished":
        # We only return a tiny result; the real artifacts are on disk under TMP_DIR
        resp["result"] = job.result
    return resp

@app.post("/plot-motif-overview", response_model=dict)  # returns {"job_id": "..."}
async def plot_motif_overview_enqueue(
    window: int = Form(500),
    data_id: str = Form(...),
    per_motif_pvals_json: str = Form(default=""),
    download: str = Form("false"),
):
    # quick preflight so users get fast feedback
    needs = [f"{TMP_DIR}/{data_id}_peaks.pkl", f"{TMP_DIR}/{data_id}_df_hits.pkl",
             f"{TMP_DIR}/{data_id}_motif_list.pkl", f"{TMP_DIR}/{data_id}_genes_lfc.pkl"]
    for p in needs:
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"Missing cache: {os.path.basename(p)}")

    want_download = download.lower() in ("true", "1", "csv")
    job = q.enqueue(
        plot_overview_single_task,
        data_id, window, per_motif_pvals_json, want_download,
        retry=Retry(max=3, interval=[10,30,60]),
        job_timeout="3600",
    )
    return {"job_id": job.get_id()}

@app.get("/plots/overview/{data_id}")
def fetch_plot_overview(data_id: str):
    # return the JSON artifacts if they exist
    try:
        fig = json.load(open(os.path.join(TMP_DIR, f"{data_id}_overview.json")))
        ordered = json.load(open(os.path.join(TMP_DIR, f"{data_id}_ordered_peaks.json")))
        final_hits = json.load(open(os.path.join(TMP_DIR, f"{data_id}_final_hits.json")))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifacts not ready")
    return {
        "overview_plot": fig,
        "data_id": data_id,
        "peak_list": ordered,
        "final_hits": final_hits,
    }
@app.get("/download/overview/{data_id}")
def download_overview_csv(data_id: str):
    fp = os.path.join(TMP_DIR, f"{data_id}_final_hits.csv")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="CSV not found (did you request download?)")
    return Response(
        content=open(fp, "rb").read(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{data_id}_final_hits.csv\"'}
    )

@app.post("/plot-motif-overview-compare", response_model=dict)  # returns {"job_id": "..."}
async def plot_motif_overview_compare_enqueue(
    session_id: str = Form(...),
    window: int = Form(500),
    per_motif_pvals_json: str = Form(default=""),
    download: str = Form("false"),
    merge: str = Form("true"),
):
    # preflight
    if not os.path.exists(os.path.join(TMP_DIR, f"{session_id}_motif_list.pkl")):
        raise HTTPException(status_code=404, detail="No cached motif_list for this session")

    # validate datasets have their caches
    datasets = _load_session(session_id)
    for ds in datasets:
        for suf in ("_peaks.pkl", "_df_hits.pkl", "_genes_lfc.pkl"):
            p = os.path.join(TMP_DIR, f"{ds.data_id}{suf}")
            if not os.path.exists(p):
                raise HTTPException(status_code=404, detail=f"Cache miss for {ds.label}: {os.path.basename(p)}")

    want_download = download.lower() in ("true", "1", "csv", "zip")
    merge_files  = merge.lower() in ("true", "1", "yes")

    job = q.enqueue(
        plot_overview_compare_task,
        session_id, window, per_motif_pvals_json, want_download, merge_files,
        retry=Retry(max=3, interval=[10,30,60]),
        job_timeout="7200",
    )
    return {"job_id": job.get_id()}

@app.get("/plots/overview-compare/{session_id}")
def fetch_plot_overview_compare(session_id: str):
    # collect per-label JSONs
    from glob import glob
    pattern = os.path.join(TMP_DIR, f"{session_id}_*_overview.json")
    files = glob(pattern)
    if not files:
        raise HTTPException(status_code=404, detail="Artifacts not ready")

    figures, ordered, final_hits = {}, {}, {}
    for fp in files:
        # file name: {session_id}_{label}_overview.json
        fname = os.path.basename(fp)
        label = fname[len(session_id)+1:].replace("_overview.json", "")
        figures[label] = json.load(open(fp))
        ordered[label] = json.load(open(os.path.join(TMP_DIR, f"{session_id}_{label}_ordered_peaks.json")))
        final_hits[label] = json.load(open(os.path.join(TMP_DIR, f"{session_id}_{label}_final_hits.json")))
    return {
        "session_id": session_id,
        "figures": figures,
        "ordered_peaks": ordered,
        "final_hits": final_hits,
    }

@app.get("/download/overview-compare/{session_id}")
def download_overview_compare(session_id: str, merged: bool = True):
    if merged:
        fp = os.path.join(TMP_DIR, f"{session_id}_final_hits_merged.csv")
        if not os.path.exists(fp):
            raise HTTPException(status_code=404, detail="Merged CSV not found (did you request merge+download?)")
        return Response(
            content=open(fp, "rb").read(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename=\"{session_id}_final_hits_merged.csv\"'}
        )
    else:
        fp = os.path.join(TMP_DIR, f"{session_id}_final_hits.zip")
        if not os.path.exists(fp):
            raise HTTPException(status_code=404, detail="ZIP not found (did you request split+download?)")
        return Response(
            content=open(fp, "rb").read(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename=\"{session_id}_final_hits.zip\"'}
        )


@app.post("/filter-motif-hits")
async def filter_and_score_enqueue(
    atac_bed: UploadFile = File(None),
    chip_bed: UploadFile = File(None),
    data_id: str = Form(...),
):
    # Write uploads to TMP_DIR (web thread) — then worker reads from disk.
    have_atac = have_chip = False
    if not atac_bed and not chip_bed:
        raise HTTPException(status_code=400, detail="Provide at least one of ATAC/ChIP.")

    if atac_bed:
        atac_path = os.path.join(TMP_DIR, f"{data_id}_atac.bed")
        with open(atac_path, "wb") as f: f.write(await atac_bed.read())
        have_atac = True
    if chip_bed:
        chip_path = os.path.join(TMP_DIR, f"{data_id}_chip.bed")
        with open(chip_path, "wb") as f: f.write(await chip_bed.read())
        have_chip = True

    job = q.enqueue(
        filter_score_single_task,
        data_id, have_atac, have_chip,
        retry=Retry(max=3, interval=[10,30,60]),
        job_timeout="1800",
    )
    return {"job_id": job.get_id()}

@app.post("/filter-motif-hits-batch")
async def filter_and_score_batch_enqueue(
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

    if any_shared:
        # Save once under session-level names
        if atac_bed_shared:
            with open(os.path.join(TMP_DIR, f"{session_id}_atac_shared.bed"), "wb") as f:
                f.write(await atac_bed_shared.read())
        if chip_bed_shared:
            with open(os.path.join(TMP_DIR, f"{session_id}_chip_shared.bed"), "wb") as f:
                f.write(await chip_bed_shared.read())
        mode = "shared"
    else:
        # Must have exactly 2 datasets for A/B
        if len(datasets) != 2:
            raise HTTPException(status_code=400, detail="A/B uploads require exactly two datasets.")
        # Save per-dataset (A = datasets[0], B = datasets[1])
        dsA, dsB = datasets[0].data_id, datasets[1].data_id
        if atac_bed_a:
            with open(os.path.join(TMP_DIR, f"{dsA}_atac.bed"), "wb") as f: f.write(await atac_bed_a.read())
        if chip_bed_a:
            with open(os.path.join(TMP_DIR, f"{dsA}_chip.bed"), "wb") as f: f.write(await chip_bed_a.read())
        if atac_bed_b:
            with open(os.path.join(TMP_DIR, f"{dsB}_atac.bed"), "wb") as f: f.write(await atac_bed_b.read())
        if chip_bed_b:
            with open(os.path.join(TMP_DIR, f"{dsB}_chip.bed"), "wb") as f: f.write(await chip_bed_b.read())
        mode = "ab"

    job = q.enqueue(
        filter_score_batch_task,
        session_id, mode,
        retry=Retry(max=3, interval=[10,30,60]),
        job_timeout="3600",
    )
    return {"job_id": job.get_id()}


@app.post("/plot-filtered-overview")
async def plot_filtered_overview_enqueue(
    chip: str = Form("false"),
    atac: str = Form("false"),
    window: int = Form(500),
    use_hit_number: str = Form("false"),
    use_match_score: str = Form("false"),
    chosen_motif: str = Form(""),
    best_transcript: str = Form("false"),
    data_id: str = Form(...),
    per_motif_pvals_json: str = Form(default=""),
    download: str = Form("false"),
):
    job = q.enqueue(
        plot_filtered_single_task,
        data_id,
        chip.lower()=="true",
        atac.lower()=="true",
        window,
        use_hit_number.lower()=="true",
        use_match_score.lower()=="true",
        (chosen_motif or None),
        best_transcript.lower()=="true",
        per_motif_pvals_json,
        download.lower() in ("true","1","csv"),
        retry=Retry(max=3, interval=[10,30,60]),
        job_timeout="3600",
    )
    return {"job_id": job.get_id()}

@app.get("/plots/filtered/{data_id}")
def fetch_filtered_overview(data_id: str):
    try:
        fig = json.load(open(os.path.join(TMP_DIR, f"{data_id}_filtered_overview.json")))
        ordered = json.load(open(os.path.join(TMP_DIR, f"{data_id}_filtered_ordered.json")))
        final_hits = json.load(open(os.path.join(TMP_DIR, f"{data_id}_filtered_final_hits.json")))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifacts not ready")
    return {"overview_plot": fig, "data_id": data_id, "peak_list": ordered, "final_hits": final_hits}

@app.get("/download/filtered/{data_id}")
def download_filtered_csv(data_id: str):
    fp = os.path.join(TMP_DIR, f"{data_id}_filtered_final_hits.csv")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="CSV not found (did you request download?)")
    return Response(open(fp,"rb").read(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename=\"{data_id}_filtered_final_hits.csv\"'})
@app.post("/plot-filtered-overview-compare")
async def plot_filtered_overview_compare_enqueue(
    session_id: str = Form(...),
    window: int = Form(500),
    chip: str = Form("false"),
    atac: str = Form("false"),
    use_hit_number: str = Form("false"),
    use_match_score: str = Form("false"),
    chosen_motif: str = Form(""),
    best_transcript: str = Form("false"),
    per_motif_pvals_json: str = Form(default=""),
    download: str = Form("false"),
    merge: str = Form("true"),
):
    job = q.enqueue(
        plot_filtered_compare_task,
        session_id,
        window,
        chip.lower()=="true",
        atac.lower()=="true",
        use_hit_number.lower()=="true",
        use_match_score.lower()=="true",
        (chosen_motif or None),
        best_transcript.lower()=="true",
        per_motif_pvals_json,
        download.lower() in ("true","1","csv","zip"),
        merge.lower() in ("true","1","yes"),
        retry=Retry(max=3, interval=[10,30,60]),
        job_timeout="7200",
    )
    return {"job_id": job.get_id()}

@app.get("/plots/filtered-compare/{session_id}")
def fetch_filtered_overview_compare(session_id: str):
    from glob import glob
    files = glob(os.path.join(TMP_DIR, f"{session_id}_*_filtered_overview.json"))
    if not files:
        raise HTTPException(status_code=404, detail="Artifacts not ready")
    figures, ordered, final_hits = {}, {}, {}
    for fp in files:
        fname = os.path.basename(fp)
        label = fname[len(session_id)+1:].replace("_filtered_overview.json", "")
        figures[label] = json.load(open(fp))
        ordered[label] = json.load(open(os.path.join(TMP_DIR, f"{session_id}_{label}_filtered_ordered.json")))
        final_hits[label] = json.load(open(os.path.join(TMP_DIR, f"{session_id}_{label}_filtered_final_hits.json")))
    return {"session_id": session_id, "figures": figures, "ordered_peaks": ordered, "final_hits": final_hits}

@app.get("/download/filtered-compare/{session_id}")
def download_filtered_compare(session_id: str, merged: bool = True):
    if merged:
        fp = os.path.join(TMP_DIR, f"{session_id}_filtered_final_hits_merged.csv")
        if not os.path.exists(fp):
            raise HTTPException(status_code=404, detail="Merged CSV not found (did you request merge+download?)")
        return Response(open(fp,"rb").read(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename=\"{session_id}_filtered_final_hits_merged.csv\"'})
    else:
        fp = os.path.join(TMP_DIR, f"{session_id}_filtered_final_hits.zip")
        if not os.path.exists(fp):
            raise HTTPException(status_code=404, detail="ZIP not found (did you request split+download?)")
        return Response(open(fp,"rb").read(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename=\"{session_id}_filtered_final_hits.zip\"'})


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
    #print(tf_results)
    return {"tfs": tf_results}

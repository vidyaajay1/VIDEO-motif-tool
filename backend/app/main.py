import os, uuid, shutil, pickle, json, csv, re
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request,  HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Tuple
from redis import Redis
from rq import Queue, Retry
from rq.job import Job
from app.tasks import (
    scan_session_batch, 
    scan_single_dataset,
    filter_score_single_task, 
    filter_score_batch_task,
    plot_filtered_single_task, 
    plot_filtered_compare_task,
    chip_overlay_single_task,
    chip_overlay_compare_task,
    run_streme_task)
from app.new_process_input import process_genomic_input, get_motif_list
from app.run_streme import write_fasta_from_genes, run_streme_on_fasta, parse_streme_results
from app.process_tomtom import subset_by_genes
from app.filter_tfs import filter_tfs_from_gene_list
from app.utils import clear_tmp_dir
from app.mem_logger import start_mem_logger
import app.models as models
from pathlib import Path

TMP_DIR = Path(os.getenv("VIDEO_TMP_DIR", "/app_data/tmp_jobs")).resolve()
TMP_DIR.mkdir(parents=True, exist_ok=True)



#BASE_DIR = Path(__file__).resolve().parent  # .../app
#TMP_DIR = (BASE_DIR.parent / "tmp").resolve()  # repo_root/tmp, absolute
#TMP_DIR.mkdir(parents=True, exist_ok=True)
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


###
import logging
logger = logging.getLogger("uvicorn")
logger.info(f"[VIDEO] TMP_DIR set to: {TMP_DIR}")
if not TMP_DIR.exists():
    logger.error(f"[VIDEO] TMP_DIR does not exist: {TMP_DIR}")
else:
    # Show what's currently inside
    logger.info(f"[VIDEO] TMP_DIR contents: {list(TMP_DIR.iterdir())}")
###
app.mount("/api/tmp", StaticFiles(directory=TMP_DIR), name="tmp")
app.mount("/tmp", StaticFiles(directory=TMP_DIR), name="tmp_alt") #backup compatibility mount
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
        detail="No matching genes detected. Please check your input gene list try again after reloading the page."
    )
    # Determine which input genes were matched
    def extract_gene(peak_id):
        return re.split(r"_FBtr", peak_id)[0]

    matched_genes = {extract_gene(pid) for pid in peaks_df['Peak_ID']}
    unmatched_genes = [g for g in gene_list if g not in matched_genes]

    # ————— persist and respond —————
    data_id = uuid.uuid4().hex
    peaks_df.to_pickle(os.path.join(TMP_DIR, f"{data_id}_peaks.pkl"))
    if gene_list:
        with open(os.path.join(TMP_DIR, f"{data_id}_genes.pkl"), "wb") as f:
            pickle.dump(gene_list, f)
        with open(os.path.join(TMP_DIR, f"{data_id}_genes_lfc.pkl"), "wb") as lf:
            pickle.dump(gene_lfc, lf) 
 
    peak_list = list(peaks_df['Peak_ID'])           
    peak_list = sorted(
        peak_list,
        key=lambda pid: abs(gene_lfc.get(extract_gene(pid), float("-inf")))
    )
    return models.OverviewResponse(
        genome="fasta/genome.fa",
        peaks_df=peaks_df.to_dict(orient="records"),
        data_id=data_id,
        peak_list=peak_list,
        unmatched_genes=unmatched_genes
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
    scan_job = q.enqueue(
        scan_single_dataset, data_id, window, fimo_threshold,
        job_id=f"scan_{data_id}",
        retry=Retry(max=3, interval=[10,30,60]),
        job_timeout="600"
    )
    return {"job_id": scan_job.get_id(), "data_id": data_id}

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
        job_id=f"scan_{session_id}",              # ← deterministic ID
        retry=Retry(max=3, interval=[10, 30, 60]),
        job_timeout="1200"
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
    # Chain to scan if it’s still running
    depends = None
    try:
        sj = Job.fetch(f"scan_{data_id}", connection=redis_conn)
        if sj.get_status() in ("queued", "started", "deferred"):
            depends = sj
    except Exception:
        pass

    if depends is None:
        # preflight: ensure caches from scan exist
        needs = [f"{TMP_DIR}/{data_id}_peaks.pkl", f"{TMP_DIR}/{data_id}_df_hits.pkl",
                 f"{TMP_DIR}/{data_id}_motif_list.pkl", f"{TMP_DIR}/{data_id}_genes_lfc.pkl"]
        for p in needs:
            if not os.path.exists(p):
                # 409 "still cooking" so the UI can retry
                raise HTTPException(status_code=409, detail=f"Missing cache: {os.path.basename(p)}")

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
        depends_on=depends,
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
    # Try to chain to scan if it’s still running
    depends = None
    try:
        sj = Job.fetch(f"scan_{session_id}", connection=redis_conn)   # ← match job_id above
        if sj.get_status() in ("queued", "started", "deferred"):
            depends = sj
    except Exception:
        pass

    # Build preflight needs: session-level motif list + per-dataset caches
    datasets = _load_session(session_id)
    dataset_ids = [ds.data_id for ds in datasets]

    needs = [f"{TMP_DIR}/{session_id}_motif_list.pkl"]  # this one IS session-level
    # hits come from scan_session_batch; peaks should exist from earlier step
    for dsid in dataset_ids:
        needs.extend([
            f"{TMP_DIR}/{dsid}_peaks.pkl",
            f"{TMP_DIR}/{dsid}_df_hits.pkl",
        ])

    # If your plotting requires gene LFC only when best_transcript is true,
    # check it conditionally to avoid spurious 409s.
    if best_transcript.lower() == "true":
        for dsid in dataset_ids:
            needs.append(f"{TMP_DIR}/{dsid}_genes_lfc.pkl")

    # Only fail if scan isn't running AND something is missing
    if depends is None:
        for p in needs:
            if not os.path.exists(p):
                raise HTTPException(status_code=409, detail=f"Missing cache: {os.path.basename(p)}")

    job = q.enqueue(
        plot_filtered_compare_task,
        session_id,
        window,
        chip.lower() == "true",
        atac.lower() == "true",
        use_hit_number.lower() == "true",
        use_match_score.lower() == "true",
        (chosen_motif or None),
        best_transcript.lower() == "true",
        per_motif_pvals_json,
        download.lower() in ("true","1","csv","zip"),
        merge.lower() in ("true","1","yes"),
        retry=Retry(max=3, interval=[10,30,60]),
        job_timeout="7200",
        depends_on=depends,                     # ← actually chain if scan is in-flight
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


@app.post("/plot-chip-overlay")  # enqueue
async def plot_chip_overlay_enqueue(
    data_id: str = Form(...),
    bigwigs: List[UploadFile] = File(...),
    chip_tracks: List[str] = Form(...),   # each "Label|#RRGGBB"
    gene: str = Form(...),
    window: int = Form(500),
    per_motif_pvals_json: str = Form(default=""),
    min_score_bits: float = Form(0.0),
):
    if len(bigwigs) != len(chip_tracks):
        raise HTTPException(status_code=400, detail="Mismatch in bigwig and track metadata count")

    # save uploaded .bw files once; pass paths to worker
    bw_inputs: List[Tuple[str, str, str]] = []
    for i, bw_file in enumerate(bigwigs):
        try:
            label, color = chip_tracks[i].split("|", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid track meta '{chip_tracks[i]}'. Use 'Label|#RRGGBB'.")
        path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.bw")
        with open(path, "wb") as f:
            shutil.copyfileobj(bw_file.file, f)
        bw_inputs.append((path, label.strip(), color.strip()))
        try:
            bw_file.file.close()
        except Exception:
            pass

    job = q.enqueue(
        chip_overlay_single_task,
        data_id, gene, int(window), bw_inputs, per_motif_pvals_json, float(min_score_bits),
        retry=Retry(max=2, interval=[10, 30]),
        job_timeout="3600",
    )
    return {"job_id": job.get_id()}

@app.get("/plots/chip-overlay/{data_id}/{gene}")  # fetch
def fetch_chip_overlay(data_id: str, gene: str):
    fp = os.path.join(TMP_DIR, f"{data_id}_{gene}_chip_overlay.json")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="Overlay not ready")
    return {"chip_overlay_plot": json.load(open(fp)), "data_id": data_id, "gene": gene}

@app.post("/plot-chip-overlay-compare")  # enqueue
async def plot_chip_overlay_compare_enqueue(
    session_id: str = Form(...),
    label: str = Form(...),
    bigwigs: List[UploadFile] = File(...),
    chip_tracks: List[str] = Form(...),
    gene: str = Form(...),
    window: int = Form(500),
    per_motif_pvals_json: str = Form(default=""),
    min_score_bits: float = Form(0.0),
):
    if len(bigwigs) != len(chip_tracks):
        raise HTTPException(status_code=400, detail="Mismatch in bigwig and track metadata count")

    bw_inputs: List[Tuple[str, str, str]] = []
    for i, bw_file in enumerate(bigwigs):
        try:
            tr_label, color = chip_tracks[i].split("|", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid track meta '{chip_tracks[i]}'. Use 'Label|#RRGGBB'.")
        path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.bw")
        with open(path, "wb") as f:
            shutil.copyfileobj(bw_file.file, f)
        bw_inputs.append((path, tr_label.strip(), color.strip()))
        try:
            bw_file.file.close()
        except Exception:
            pass

    job = q.enqueue(
        chip_overlay_compare_task,
        session_id, label, gene, int(window), bw_inputs, per_motif_pvals_json, float(min_score_bits),
        retry=Retry(max=2, interval=[10, 30]),
        job_timeout="3600",
    )
    return {"job_id": job.get_id()}

@app.get("/plots/chip-overlay-compare/{session_id}/{label}/{gene}")  # fetch
def fetch_chip_overlay_compare(session_id: str, label: str, gene: str):
    fp = os.path.join(TMP_DIR, f"{session_id}_{label}_{gene}_chip_overlay.json")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="Overlay not ready")
    return {"session_id": session_id, "label": label, "gene": gene, "chip_overlay_plot": json.load(open(fp))}


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
    gene_file: Optional[UploadFile] = File(None),
):
    try:
        genome_path = "fasta/genome.fa"

        # --- Build gene_list in the API process ---
        if use_de_genes:
            if not tissue or not stage:
                raise HTTPException(status_code=400, detail="Missing tissue or stage selection")

            sheet_map = {
                "10-12": "stage 10-12",
                "13-16": "stage 13-16",
            }
            if stage not in sheet_map:
                raise HTTPException(status_code=400, detail="Invalid stage")

            sheet_name = sheet_map[stage]
            if sheet_name not in TF_DATA_CACHE:
                raise HTTPException(status_code=404, detail="Stage data not loaded")

            df = TF_DATA_CACHE[sheet_name]
            if "gene" not in df.columns or "cell_types" not in df.columns:
                raise HTTPException(status_code=500, detail="Required columns missing in DE data")

            gene_list = df[df["cell_types"] == tissue]["gene"].dropna().head(250).tolist()
            if not gene_list:
                raise HTTPException(status_code=404, detail="No DE genes found for this tissue")
        else:
            if gene_file is None:
                raise HTTPException(status_code=400, detail="Gene list file missing")
            df = pd.read_csv(gene_file.file)
            if df.empty or df.shape[1] < 1:
                raise HTTPException(status_code=400, detail="Uploaded gene file is empty or invalid")
            gene_list = df.iloc[:, 0].dropna().tolist()
            if not gene_list:
                raise HTTPException(status_code=400, detail="No genes found in the uploaded file")

        # --- Enqueue the background job ---
        tmp_id = uuid.uuid4().hex
        job = q.enqueue(
            run_streme_task,
            gene_list,
            minw,
            maxw,
            window_size,
            tmp_id,
            genome_path,           # keyword optional; matches the default anyway
            job_timeout="3600",     # optional per-job override
        )

        # Return identifiers the client can use to poll & later download
        return {
            "job_id": job.get_id(),
            "tmp_id": tmp_id,
            "status": "queued",
            "poll_url": f"/jobs/{job.get_id()}",
        }

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return JSONResponse(content={"error": f"Job not found: {job_id}"}, status_code=404)

    status = job.get_status()
    progress = job.meta.get("progress", 0)
    if status == "finished":
        # result contains {"motifs", "streme_html_url", "tmp_id"}
        return {"status": status, "progress": 100, "result": job.result}
    if status == "failed":
        return {"status": status, "progress": progress, "error": str(job.exc_info)}
    return {"status": status, "progress": progress}


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

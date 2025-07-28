import os, uuid, shutil, pickle, json, csv, re
import pandas as pd
from functools import reduce
from operator import and_
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request,  HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, AnyUrl
from app.new_process_input import process_genomic_input, get_motif_list
from app.new_scan_motifs import scan_wrapper
from app.filter_motifs import filter_motif_hits
from app.integrated_scoring import score_and_merge, score_hit_naive_bayes
from app.plotting import plot_occurence_overview, rank_peaks_for_plot
from app.bigwig_overlay import fetch_bw_coverage, plot_chip_overlay
from app.run_streme import write_fasta_from_genes, run_streme_on_fasta, parse_streme_results
from app.process_tomtom import subset_by_genes
from app.filter_tfs import filter_tfs_from_gene_list

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

    yield

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
class PlotOverviewResponse(BaseModel):
    overview_image: AnyUrl
    data_id: str
    peak_list: List[str] 
class ScannerResponse(BaseModel):
    data_id: str
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

    # run your get_motif_list (this will raise ValueError on any bad motif)
    try:
        motif_list = get_motif_list(motif_inputs)
        with open(f"{TMP_DIR}/{data_id}_motif_list.pkl", "wb") as f:
            pickle.dump(motif_list, f)
    except ValueError as e:
        # send back exactly what failed
        raise HTTPException(status_code=400, detail=str(e))

    # if you like, cache it for this session / data_id; or just return success
    return {"status": "ok", "count": len(motif_list)}


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

@app.post("/plot-motif-overview", response_model=PlotOverviewResponse)
async def plot_motif_overview(
    request: Request,
    window: int = Form(500),
    data_id: str = Form(...)
):
 
    # 2) Load the peaks from the earlier get-genomic-input step
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
        raise HTTPException(status_code=404, detail="No cached motif_hits for given data_id")
    
    try:
        gene_lfc = pd.read_pickle(f"{TMP_DIR}/{data_id}_genes_lfc.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached gene_logfc dict for given data_id")

    # 5) Plot & return
    out_png = os.path.join(TMP_DIR, f"{uuid.uuid4()}.png")
    peak_list = list(peaks_df['Peak_ID'])
    peak_ranks = rank_peaks_for_plot(
    df_hits=df_hits,
    gene_lfc=gene_lfc,
    peaks_df=peaks_df,
    use_hit_number=False,
    use_match_score=False,
    motif=None,
)
    plot_path, peak_list = plot_occurence_overview(
        peak_list=peak_list,
        peaks_df=peaks_df,
        motifs=motif_list,
        df_hits=df_hits,
        window=window,
        peak_rank = peak_ranks,
        output_path=out_png
    )
    url = request.url_for("tmp", path=os.path.basename(plot_path))
    return {
        "overview_image": str(url),
        "data_id": data_id,
        "peak_list": peak_list
    }

@app.post("/filter-motif-hits", response_model=ScannerResponse)
async def filter_and_score(
    atac_bed: UploadFile = File(None),      # now optional
    chip_bed: UploadFile = File(None),      # now optional
    data_id: str = Form(...),
):
    # 1) At least one filter type must be provided
    if not atac_bed and not chip_bed:
        raise HTTPException(
            status_code=400,
            detail="You must provide at least one of ATAC or ChIP BED files."
        )

    # 2) Load your cached motif‐hits
    try:
        with open(f"{TMP_DIR}/{data_id}_df_hits.pkl","rb") as f:
            df_hits = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Missing cache for data_id")

    # Helper to make an *empty* hits‐DataFrame
    empty_hits = pd.DataFrame(columns=df_hits.columns)

    # 3) If the user sent an ATAC file, save and filter; otherwise use empty
    if atac_bed:
        atac_path = os.path.join(TMP_DIR, f"{data_id}_atac.bed")
        with open(atac_path,"wb") as f:
            f.write(await atac_bed.read())
        atac_filt_hits = filter_motif_hits(df_hits, atac_path)
        with open(f"{TMP_DIR}/{data_id}_atac_filt_hits.pkl","wb") as f:
            pickle.dump(atac_filt_hits, f)
    else:
        atac_filt_hits = empty_hits

    # 4) Ditto for ChIP
    if chip_bed:
        chip_path = os.path.join(TMP_DIR, f"{data_id}_chip.bed")
        with open(chip_path,"wb") as f:
            f.write(await chip_bed.read())
        chip_filt_hits = filter_motif_hits(df_hits, chip_path)
        with open(f"{TMP_DIR}/{data_id}_chip_filt_hits.pkl","wb") as f:
            pickle.dump(chip_filt_hits, f)
    else:
        chip_filt_hits = empty_hits

    # 5) Now run your scorer.  Because chip_filt_hits or atac_filt_hits might be empty,
    #    raw_scoring_chip/atac will end up generating zero‐counts for all (Peak_ID, Motif).
    scored_df = score_and_merge(df_hits, chip_filt_hits, atac_filt_hits)

    # 6) And the rest of your pipeline is unchanged
    scored_df.to_csv("scored_df.csv")
    ranked_df = score_hit_naive_bayes(scored_df)
    top_hits = (
        ranked_df
        .reset_index()
        .sort_values("P_regulatory", ascending=False)
        [["Peak_ID", "Motif", "P_regulatory", "M_prom", "M_chip", "M_atac", "logFC", "FIMO_score"]]
    )
    top_hits.to_csv(f"{TMP_DIR}/{data_id}_top_hits.tsv", sep="\t", index=False)

    return {"data_id": data_id}


@app.get("/download-top-hits/{data_id}")
async def download_top_hits(data_id: str):
    file_path = os.path.join(TMP_DIR, f"{data_id}_top_hits.tsv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Top hits file not found")

    return FileResponse(
        file_path,
        media_type="text/tab-separated-values",
        filename="top_predicted_regulatory_hits.tsv"
    )


def load_filtered_hits(data_id: str, modality: str) -> pd.DataFrame:
    path = os.path.join(TMP_DIR, f"{data_id}_{modality}_filt_hits.pkl")
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        # return empty DF with expected columns rather than 404
        return pd.DataFrame(columns=["Peak_ID", "Motif", "Score_bits", "Rel_pos"])

@app.post("/plot-filtered-overview", response_model=PlotOverviewResponse)
async def plot_filtered_overview(
    request: Request,
    chip: str = Form("false"),
    atac: str = Form("false"),
    window: int = Form(500),
    use_hit_number: str = Form("false"),
    use_match_score: str = Form("false"),
    chosen_motif: str = Form(...),
    data_id: str = Form(...)
):
    # parse booleans
    chip             = chip.lower() == "true"
    atac             = atac.lower() == "true"
    use_hit_number   = use_hit_number.lower() == "true"
    use_match_score  = use_match_score.lower() == "true"

    # ——— 1) Load base data ———
    try:
        peaks_df   = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_peaks.pkl"))
        gene_lfc   = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_genes_lfc.pkl"))
        motif_list = pd.read_pickle(os.path.join(TMP_DIR, f"{data_id}_motif_list.pkl"))
    except FileNotFoundError:
        raise HTTPException(404, "No cached genomic data for given data_id")

    # ——— 3) Build df_to_plot ———
    selections = []
    if chip:  
        selections.append("chip")
    if atac:  
        selections.append("atac")

    if not selections:
        # no modality filter → use the raw union of both ChIP & ATAC hit
        df_to_plot = pd.read_pickle(f"{TMP_DIR}/{data_id}_df_hits.pkl")
        # default to sorting by hit‐number when no explicit flag

    else:
        dfs = [load_filtered_hits(data_id, mod) for mod in selections]
        df_to_plot = (
            reduce(lambda L, R: pd.merge(L, R, how="inner"), dfs)
            if len(dfs) > 1 else dfs[0]
        )

    # ——— 4) Compute peak ranks ———
    # pass an empty DF if df_to_plot is None so the function sees a DataFrame type
    peak_ranks = rank_peaks_for_plot(
        df_hits         = df_to_plot,
        gene_lfc        = gene_lfc,
        peaks_df        = peaks_df,
        use_hit_number  = use_hit_number,
        use_match_score = use_match_score,
        motif           = chosen_motif,
    )

    # ——— 5) Generate plot ———
    out_png   = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.png")
    peak_list = list(peaks_df["Peak_ID"])
    plot_path, peak_list = plot_occurence_overview(
        peak_list   = peak_list,
        peaks_df    = peaks_df,
        motifs      = motif_list,
        df_hits     = df_to_plot,    # None if no filtering
        window      = window,
        peak_rank   = peak_ranks,
        output_path = out_png,
    )

    url = request.url_for("tmp", path=os.path.basename(plot_path))
    return {
        "overview_image": str(url),
        "data_id": data_id,
        "peak_list": peak_list,
    }

@app.post("/plot-chip-overlay", response_class=FileResponse)
async def plot_chip_overlay_route(
    data_id: str = Form(...),
    bigwigs: List[UploadFile] = File(...),
    chip_tracks: List[str] = Form(...),
    gene: str = Form(...),
    window: int = Form(500)
):
    temp_dir = "tmp"

    # Load cached motif data
    peaks_df = pd.read_pickle(f"{temp_dir}/{data_id}_peaks.pkl")
    with open(f"{temp_dir}/{data_id}_motif_list.pkl", "rb") as f:
        motif_list = pickle.load(f)
    with open(f"{temp_dir}/{data_id}_motif_hits.pkl", "rb") as f:
        motif_hits = pickle.load(f)

    # Save bigWigs
    if len(bigwigs) != len(chip_tracks):
        return {"error": "Mismatch in bigwig and track metadata count"}

    bw_input = []
    for i, bw_file in enumerate(bigwigs):
        label, color = chip_tracks[i].split("|")
        bw_path = os.path.join(temp_dir, f"{uuid.uuid4()}.bw")
        with open(bw_path, "wb") as f:
            shutil.copyfileobj(bw_file.file, f)
        bw_input.append((bw_path, label, color))

    # Plot
    coverage_data = fetch_bw_coverage(gene, peaks_df, bw_input)
    if not coverage_data:
        return {"error": f"No coverage data for {gene}"}

    output_path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
    plot_chip_overlay(coverage_data, motif_list, motif_hits, gene, window, output_path)

    return FileResponse(output_path, media_type="image/png")

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

            gene_list = (
                df[df["cell_types"] == tissue]["gene"].dropna().head(250).tolist()
            )

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

        gene_list = (
            df[df["cell_types"] == tissue]["gene"].dropna().head(250).tolist()
        )

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


from fastapi import FastAPI, UploadFile, File, Form, Request,  HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import shutil, os, uuid, pickle, subprocess, json, pandas as pd
from app.utils import reverse_complement
from pyfaidx import Fasta
from app.process_motifs import scan_peaks_for_motifs, process_genomic_input, filter_motif_hits, get_motif_list, generate_plot
from app.scanner import motif_to_memefile, peaks_df_to_fasta, run_fimo
from app.bigwig_overlay import fetch_bw_coverage, plot_chip_overlay
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
from pydantic import BaseModel, AnyUrl
from contextlib import asynccontextmanager


TMP_DIR = "tmp"

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
    yield

# pass the lifespan function into FastAPI()
app = FastAPI(lifespan=lifespan)

# ensure the directory exists, then mount it
os.makedirs(TMP_DIR, exist_ok=True)
app.mount("/tmp", StaticFiles(directory=TMP_DIR), name="tmp")

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
class PlotOverviewResponse(BaseModel):
    overview_image: AnyUrl
    data_id: str
    peak_list: List[str]  
@app.post("/get-genomic-input", response_model=OverviewResponse)
async def get_genomic_input(
    request: Request,
    bed_file: Optional[UploadFile] = File(None),
    gene_list_file: UploadFile = File(None),
    window_size: int = Form(500),
):
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)

    if bed_file is None and gene_list_file is None:
        return JSONResponse({"error": "Provide a gene list or a BED file"}, status_code=400)
    
    bed_path = None
    if bed_file is not None:
        bed_path = os.path.join(temp_dir, f"{uuid.uuid4()}.bed")
        with open(bed_path, "wb") as f:
            shutil.copyfileobj(bed_file.file, f)
    
    gene_list = None
    if gene_list_file:
        gene_list_path = os.path.join(temp_dir, f"{uuid.uuid4()}.csv")
        with open(gene_list_path, "wb") as f:
            shutil.copyfileobj(gene_list_file.file, f)
        with open(gene_list_path, encoding="utf-8-sig") as f:
            gene_list = [line.strip() for line in f if line.strip()]
        
    _, peaks_df = process_genomic_input(
        genome_filepath="fasta/genome.fa",
        gtf_filepath="fasta/dmel_genes.gtf",
        bed_path=bed_path,  
        window_size=window_size,
        gene_list=gene_list
    )

    data_id = str(uuid.uuid4())
    peaks_df.to_pickle(f"tmp/{data_id}_peaks.pkl")
    if gene_list:
        with open(f"tmp/{data_id}_genes.pkl", "wb") as f:
            pickle.dump(gene_list, f)
    
    return OverviewResponse(
        genome="fasta/genome.fa",
        peaks_df=peaks_df.to_dict(orient="records"),
        data_id=data_id
    )

@app.post("/run-streme")
async def run_streme(request: Request, data_id: str = Form(...)):
    import xml.etree.ElementTree as ET

    try:
        peaks_df = pd.read_pickle(f"tmp/{data_id}_peaks.pkl")
        genome = Fasta("fasta/genome.fa")
    except Exception as e:
        return JSONResponse({"error": f"Failed to load cached data: {e}"}, status_code=500)

    # Extract sequences
    output_id = str(uuid.uuid4())
    fasta_path = f"tmp/{output_id}_peaks.fa"
    with open(fasta_path, "w") as f:
        for i, row in peaks_df.iterrows():
            chrom = row["Chromosome"]
            start = int(row["Start"])
            end = int(row["End"])
            strand = row.get("Strand", "+")
            name = row.get("Gene Symbol", f"peak_{i}")
            try:
                seq = genome[chrom][start:end].seq
                if strand == "-":
                    seq = reverse_complement(seq)
                f.write(f">{name}\n{seq}\n")
            except Exception:
                continue

    # Run STREME
    streme_out = f"tmp/{output_id}_streme_out"
    os.makedirs(streme_out, exist_ok=True)
    try:
        subprocess.run([
            "streme",
            "--oc", streme_out,
            "--dna",
            "--p", fasta_path,
            "--minw", "6",
            "--maxw", "10",
            "--verbosity", "2"
        ], check=True)
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"STREME failed: {e}"}, status_code=500)

    # Parse streme.xml for motif data
    xml_path = os.path.join(streme_out, "streme.xml")
    motifs = []
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for i, motif_elem in enumerate(root.findall(".//motif")):
            consensus = motif_elem.attrib.get("id", "N/A")
            name = motif_elem.attrib.get("alt", f"STREME-{i+1}")
            pval = motif_elem.attrib.get("test_pvalue", "N/A")
            eval_ = motif_elem.attrib.get("test_evalue", "N/A")
            sites = motif_elem.attrib.get("npassing", "N/A")

            # Parse PWM from <pos> elements
            rows = motif_elem.findall("pos")
            pwm = []
            for pos_elem in rows:
                row = [
                    float(pos_elem.attrib.get("A", 0.0)),
                    float(pos_elem.attrib.get("C", 0.0)),
                    float(pos_elem.attrib.get("G", 0.0)),
                    float(pos_elem.attrib.get("T", 0.0))
                ]
                pwm.append(row)

            motifs.append({
                "id": f"{i+1}",  # 1-based index
                "consensus": consensus,
                "name": name,
                "pvalue": pval,
                "evalue": eval_,
                "nsites": sites,
                "pwm": pwm,
                "html_link": f"/tmp/{output_id}_streme_out/streme.html"  # optional
            })
    # Provide link to STREME HTML report
    html_name = f"{output_id}_streme_out/streme.html"
    html_url = request.url_for("tmp", path=html_name)

    return {
        "motifs": motifs,
        "streme_html_url": str(html_url)
    }


@app.post("/plot-motif-overview", response_model=PlotOverviewResponse)
async def initial_motif_overview(
    request: Request,
    motifs: list[str] = Form(...),
    min_score: float = Form(9.0),
    window: int = Form(500),
    data_id: str = Form(...)
):
    # 1) Validate inputs
    if not motifs:
        raise HTTPException(status_code=400, detail="At least one motif is required")

    try:
        motif_inputs = [json.loads(m) for m in motifs]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid motif JSON: {e}")

    # 2) Load the peaks from the earlier get-genomic-input step
    try:
        peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached peaks for given data_id")

    # 3) Build Motif objects & scan
    motif_list = get_motif_list(motif_inputs)
    motif_to_memefile(motif_list, "memefile1.txt")
    peaks_df_to_fasta(peaks_df, "fasta/genome.fa", "sequences.txt", window)
    run_fimo("memefile1.txt", "sequences.txt", "bed", 0.05)
    hits = scan_peaks_for_motifs(
        genome=Fasta("fasta/genome.fa"),
        peaks_df=peaks_df,
        window=window,
        motif_list=motif_list,
        min_score=min_score
    )

    with open(f"{TMP_DIR}/{data_id}_motifs.pkl", "wb") as f:
        pickle.dump(motif_list, f)
    with open(f"{TMP_DIR}/{data_id}_hits.pkl", "wb") as f:
        pickle.dump(hits, f)

    # 5) Plot & return
    out_png = os.path.join(TMP_DIR, f"{uuid.uuid4()}.png")
    plot_path, peak_list, max_score = generate_plot(
        peaks_df=peaks_df,
        motif_list=motif_list,
        motif_hits=hits,
        window_size=window,
        output_path=out_png
    )
    url = request.url_for("tmp", path=os.path.basename(plot_path))
    return {
        "overview_image": str(url),
        "data_id": data_id,
        "peak_list": peak_list,
        "max_motif_score": max_score,
    }


@app.post("/filter-motif-overview", response_model=PlotOverviewResponse)
async def filter_motif_overview(
    request: Request,
    atac_bed: UploadFile = File(...),
    data_id: str = Form(...),
    window: int = Form(500),
):
    # 1) Validate
    if not atac_bed:
        raise HTTPException(status_code=400, detail="ATAC/ChIP BED file is required")
    # 2) Load cached peaks, motifs & hits
    try:
        peaks_df = pd.read_pickle(f"{TMP_DIR}/{data_id}_peaks.pkl")
        with open(f"{TMP_DIR}/{data_id}_motifs.pkl","rb") as f: motif_list = pickle.load(f)
        with open(f"{TMP_DIR}/{data_id}_hits.pkl","rb")   as f: hits = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Missing cache for data_id")

    # 3) Save the uploaded BED
    atac_path = f"{TMP_DIR}/{data_id}_atac.bed"
    with open(atac_path,"wb") as f:
        f.write(await atac_bed.read())

    # 4) Filter
    filtered_hits = filter_motif_hits(hits, atac_path)
    with open(f"{TMP_DIR}/{data_id}_filt_hits.pkl","wb") as f:
        pickle.dump(filtered_hits, f)

    # 5) Plot & return
    print("before hits:\n",hits)
    print("filtered_hits\n" ,filtered_hits)
    out_png = os.path.join(TMP_DIR, f"{uuid.uuid4()}.png")
    plot_path, peak_list, max_score = generate_plot(
        peaks_df=peaks_df,
        motif_list=motif_list,
        motif_hits=filtered_hits,
        window_size=window,
        output_path=out_png
    )
    url = request.url_for("tmp", path=os.path.basename(plot_path))

    return {
        "overview_image": str(url),
        "data_id": data_id,
        "peak_list": peak_list,
        "max_motif_score": max_score,
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
    with open(f"{temp_dir}/{data_id}_motifs.pkl", "rb") as f:
        motif_list = pickle.load(f)
    with open(f"{temp_dir}/{data_id}_hits.pkl", "rb") as f:
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
    fig = plot_chip_overlay(coverage_data, motif_list, motif_hits, gene, window)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return FileResponse(output_path, media_type="image/png")



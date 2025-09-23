import os, shutil, uuid, subprocess
from pyfaidx import Fasta
from fastapi import HTTPException
from app.new_process_input import process_genomic_input
from app.utils import reverse_complement
import xml.etree.ElementTree as ET
from app.main import TMP_DIR 
from pathlib import Path
STREME_PATH = "/home/ec2-user/miniconda3/envs/memesuite/bin/streme"

STREME_ENV_VARS = ("VIDEO_STREME_PATH", "STREME_PATH")  # allow either
def resolve_streme_exe():
    for k in ("VIDEO_STREME_PATH", "STREME_PATH"):
        p = os.getenv(k)
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    p = shutil.which("streme")
    if p:
        return p
    raise HTTPException(
        status_code=500,
        detail="STREME binary not found. Set VIDEO_STREME_PATH/STREME_PATH "
               "or install MEME Suite (conda install -c bioconda meme)."
    )

def write_fasta_from_genes(gene_list, genome_fasta_path, window_size = 500):
    genome = Fasta(genome_fasta_path)
    # Implement your logic to write sequences based on gene list
    peaks_df = process_genomic_input(
        genome_filepath="fasta/genome.fa",
        gtf_filepath="fasta/dmel_genes.gtf",
        bed_path=None,
        window_size=window_size,
        gene_list=gene_list,
        gene_lfc = None
    )
    # For now, this is a placeholder that writes dummy entries
    # Extract sequences
    output_id = str(uuid.uuid4())
    fasta_path = str(Path(TMP_DIR) / f"{output_id}_tffinder_peaks.fa")
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
    return fasta_path

def run_streme_on_fasta(input_fasta, minw, maxw, tmp_dir):
    tmp_dir = os.path.abspath(tmp_dir)
    streme_bin = resolve_streme_exe()
    print(f"[STREME] using: {streme_bin}")
    input_fasta = os.path.abspath(input_fasta)
    output_id = str(uuid.uuid4())
    streme_out = os.path.abspath(os.path.join(tmp_dir, f"{output_id}_streme_out"))
    os.makedirs(streme_out, exist_ok=True)

    try:
        subprocess.run([
            streme_bin,
            "--oc", streme_out,
            "--dna",
            "--p", input_fasta,
            "--minw", str(minw),
            "--maxw", str(maxw),
            "--verbosity", "2"
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"STREME failed: {e}")

    return streme_out, output_id

def parse_streme_results(streme_out, output_id, request):
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

            pwm = []
            for pos_elem in motif_elem.findall("pos"):
                pwm.append([
                    float(pos_elem.attrib.get("A", 0.0)),
                    float(pos_elem.attrib.get("C", 0.0)),
                    float(pos_elem.attrib.get("G", 0.0)),
                    float(pos_elem.attrib.get("T", 0.0))
                ])
            html_url = request.url_for("tmp", path=f"{output_id}_streme_out/streme.html")
            motifs.append({
                "id": f"{i+1}",
                "consensus": consensus,
                "name": name,
                "pvalue": pval,
                "evalue": eval_,
                "nsites": sites,
                "pwm": pwm,
                "html_link": str(html_url),
            })

    html_url = request.url_for("tmp", path=f"{output_id}_streme_out/streme.html")
    return motifs, str(html_url)

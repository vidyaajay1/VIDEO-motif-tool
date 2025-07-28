import pandas as pd
import numpy as np
from pyfaidx import Fasta
from app.utils import iupac_to_motif
from typing import Optional, Dict

#---------------------------------------------#
# Read input GENE BED file -> peaks_df
def read_gene_bed(bed_filepath: str) -> pd.DataFrame:
    bed_dict = {}
    with open(bed_filepath, "r") as bed:
        for line in bed:
            chrom, start, end, _, _ = line.strip().split("\t")
            start = int(start)
            end = int(end)
            peak_name = f"{chrom}:{start}"
            midpoint = (start + end) // 2
            bed_dict[peak_name] = {
                "Chromosome": chrom,
                "Start": start,
                "End": end,
                "Peak Length": end - start,
                "Midpoint": midpoint
            }
    bed_df = pd.DataFrame.from_dict(bed_dict, orient="index").reset_index()
    bed_df = bed_df.rename(columns={"index": "Peak_ID"})
    
    if bed_df["Chromosome"].iloc[0].startswith("chr"):
        bed_df["Chromosome_chr"] = bed_df["Chromosome"].copy()
        bed_df["Chromosome"] = bed_df["Chromosome"].str.replace("chr", "", regex=False)
    else:
        bed_df["Chromosome_chr"] = "chr" + bed_df["Chromosome"] 
    # Filter for valid Drosophila chromosomes
    valid_chromosomes = ["2L", "2R", "3L", "3R", "4", "X", "Y", "mitochondrion_genome"]
    bed_df = bed_df[bed_df["Chromosome"].isin(valid_chromosomes)].reset_index(drop=True)
    return bed_df

#OR, if input is a gene list
# read reference GTF file
def read_gtf_all_tss(gtf_fp):
    gtf = pd.read_csv(
        gtf_fp, sep="\t", comment="#", header=None,
        names=["Chromosome","Source","Feature","Start","End","Score","Strand","Frame","Attr"]
    )
    # pull gene_symbol + transcript_id
    gtf["Gene Symbol"]   = gtf.Attr.str.extract(r'gene_symbol "([^"]+)"')
    gtf["Flybase ID"]   = gtf.Attr.str.extract(r'gene_id "([^"]+)"')
    gtf["Transcript ID"] = gtf.Attr.str.extract(r'transcript_id "([^"]+)"')

    # one row per isoform
    m = gtf.Feature == "mRNA"
    txs = gtf.loc[m, ["Chromosome","Start","End","Strand","Gene Symbol","Flybase ID", "Transcript ID"]].copy()
    txs["TSS"] = txs.apply(lambda r: r.Start if r.Strand=="+" else r.End, axis=1)

    # drop any transcripts that have the same TSS for the same gene
    txs_unique = txs.drop_duplicates(subset=["Gene Symbol", "TSS"])

    return txs_unique

# subset to genes of interest and promoter windows
def get_peaks_df_for_transcripts(
    gene_list: list[str],
    tss_df: pd.DataFrame,
    gene_lfc: Optional[Dict[str, float]],
    window: int = 500,
) -> pd.DataFrame:
    if gene_list and all(gene.startswith("FBgn") for gene in gene_list):
        df = tss_df[tss_df["Flybase ID"].isin(gene_list)].copy()
    else:
        df = tss_df[tss_df["Gene Symbol"].isin(gene_list)].copy()

    if df.empty:
        return pd.DataFrame()
    bed_records = []
    for _, row in df.iterrows():
        gene = row["Gene Symbol"]
        transcript = row["Transcript ID"]
        chrom = row["Chromosome"]
        midpoint = int(row["TSS"])
        start = midpoint - window
        end = midpoint + window
        peak_id = f"{gene}_{transcript}"

        bed_records.append({
            "Peak_ID": peak_id,
            "Chromosome": chrom,
            "Start": start,
            "End": end,
            "Peak Length": end - start,
            "Midpoint": midpoint,
            "Strand": row["Strand"],
            "logFC": gene_lfc.get(gene) if gene_lfc else None
        })

    bed_df = pd.DataFrame(bed_records)
    bed_df["Chromosome_chr"] = "chr" + bed_df["Chromosome"].astype(str)
    return bed_df

#wrapper function to process genomic input - CALL THIS FROM MAIN.PY
def process_genomic_input(
    genome_filepath: str,
    gtf_filepath: str,
    bed_path: str,  
    window_size: int,
    gene_list: list[str] = None,
    gene_lfc: Optional[Dict[str, float]] = None,
)-> pd.DataFrame:
    genome = Fasta(genome_filepath)
    all_genes_df = read_gtf_all_tss(gtf_filepath)
    if gene_list:
        peaks_df = get_peaks_df_for_transcripts(gene_list, all_genes_df, gene_lfc, window_size)
    else:
        peaks_df = read_gene_bed(bed_path)
    return peaks_df

#---------------------------------------------#
class Motif:
    def __init__(
        self,
        name: str,
        color: str,
        pwm: np.ndarray,               # shape = (L, 4)
        background: np.ndarray = None,     # length‐4 vector
        pseudocount: float = 1e-7,
    ):
        self.name = name
        self.color = color
        self.pwm = pwm + pseudocount
        self.background = background if background is not None else np.ones(4)/4
        self.log_odds = np.log2(self.pwm / self.background)
        self.length = self.log_odds.shape[0]
        self.max_score = float(np.sum(self.log_odds.max(axis=1)))

def get_motif_list(motif_inputs):
    motif_list: list[Motif] = []
    for entry in motif_inputs:
        typ   = entry["type"]
        if typ == "iupac":
           data = entry["iupac"]
        elif typ == "pwm":
           data = entry["pwm"]
        elif typ == "pcm":
           data = entry["pcm"]
        else:
           raise ValueError(f"Unknown motif type {typ!r}")
        name  = entry["name"]
        color = entry["color"]
        if typ == "iupac":
            try:
                pwm = iupac_to_motif(data)
            except Exception as e:
                raise ValueError(f"Invalid IUPAC {data}: {e}")
        elif typ == "pwm":
            # validate shape / values
            arr = np.array(data, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(f"PWM for {name} must be N×4")
            pwm = arr
        elif typ == "pcm":
            arr = np.array(data, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(f"PWM for {name} must be Nx4")
            row_sums = arr.sum(axis=1, keepdims=True)
            if np.any(row_sums == 0):
                raise ValueError(f"PCM for {name} contains a row with zero total counts")
            # normalize each row to get probabilities
            pwm = arr / row_sums
            print(pwm)
        else:
            raise ValueError(f"Unknown motif type {typ}")
        m = Motif(name, color, pwm)
        motif_list.append(m)

    if not motif_list:
        raise ValueError("No valid motifs provided")
    return motif_list
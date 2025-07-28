import pandas as pd
import numpy as np
from pyfaidx import Fasta
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Tuple, Optional, Dict
from app.utils import one_hot_encode, iupac_to_motif, reverse_complement, compute_max_motif_score
from app.plotting import plot_occurence_overview
#import pickle, uuid, os, datetime
from collections import defaultdict
from pybedtools import BedTool

#part 1: read bed file OR genes into df
def read_bed(bed_filepath: str) -> pd.DataFrame:
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

def read_gtf_all_tss(gtf_fp):
    gtf = pd.read_csv(
        gtf_fp, sep="\t", comment="#", header=None,
        names=["Chromosome","Source","Feature","Start","End","Score","Strand","Frame","Attr"]
    )
    # pull gene_symbol + transcript_id
    gtf["Gene Symbol"]   = gtf.Attr.str.extract(r'gene_symbol "([^"]+)"')
    gtf["Transcript ID"] = gtf.Attr.str.extract(r'transcript_id "([^"]+)"')

    # one row per isoform
    m = gtf.Feature == "mRNA"
    txs = gtf.loc[m, ["Chromosome","Start","End","Strand","Gene Symbol","Transcript ID"]].copy()
    txs["TSS"] = txs.apply(lambda r: r.Start if r.Strand=="+" else r.End, axis=1)

    # drop any transcripts that have the same TSS for the same gene
    txs_unique = txs.drop_duplicates(subset=["Gene Symbol", "TSS"])

    return txs_unique
def get_peaks_df_for_transcripts(
    gene_list: list[str],
    tss_df: pd.DataFrame,
    window: int = 500
) -> pd.DataFrame:
    # keep only the transcripts for genes you care about
    df = tss_df[tss_df["Gene Symbol"].isin(gene_list)].copy()
    if df.empty:
        return pd.DataFrame(), None

    bed_records = []
    for _, row in df.iterrows():
        chrom    = row["Chromosome"]
        midpoint = int(row["TSS"])
        start    = midpoint - window
        end      = midpoint + window
        
        # build an informative Peak_ID
        peak_id = f"{row['Gene Symbol']}_{row['Transcript ID']}"
        
        bed_records.append({
            "Peak_ID":        peak_id,
            "Chromosome":     chrom,
            "Start":          start,
            "End":            end,
            "Peak Length":    end - start,
            "Midpoint":       midpoint,
            "Strand": row["Strand"]
        })

    bed_df = pd.DataFrame(bed_records)
    bed_df["Chromosome_chr"] = "chr" + bed_df["Chromosome"].astype(str)
    return bed_df

#part 2: process genomic_input
def process_genomic_input(
    genome_filepath: str,
    gtf_filepath: str,
    bed_path: str,  
    window_size: int,
    gene_list: list[str] = None
):
    genome = Fasta(genome_filepath)
    all_genes_df = read_gtf_all_tss(gtf_filepath)
    if gene_list:
        peaks_df = get_peaks_df_for_transcripts(gene_list, all_genes_df, window_size)
    else:
        peaks_df = read_bed(bed_path)
    return genome, peaks_df

#part 3: get motifs
class Motif:
    def __init__(
        self,
        name: str,
        color: str,
        pwm: np.ndarray,               # shape = (L, 4)
        background: np.ndarray = None,     # length‐4 vector
        pseudocount: float = 1e-7,
        threshold_frac: float = 0.8
    ):
        self.name = name
        self.color = color
        self.pwm = pwm + pseudocount
        self.pwm_revcomp = self.pwm[::-1, [3, 2, 1, 0]] #flip rows from bottom to top to reverse and then re-index columns to complement
        self.background = background if background is not None else np.ones(4)/4
        self.log_odds = np.log2(self.pwm / self.background)
        self.log_odds_revcomp = np.log2(self.pwm_revcomp / self.background)
        self.length = self.log_odds.shape[0]
        self.max_score = float(np.sum(self.log_odds.max(axis=1)))
        self.threshold_frac = threshold_frac
        self.threshold = threshold_frac * self.max_score

    def scan_array(self, S: np.ndarray, upstream: int, revcomp = False) -> list[tuple[int, float]]:
        hits: list[tuple[int, float]] = []
        L = self.length
        N = S.shape[0]
        if revcomp == False:
            for i in range(N - L + 1):
                # elementwise multiply (L×4) * (L×4) and sum → one float
                score = float(np.sum(S[i : i + L] * self.log_odds))
                if score >= self.threshold:
                    rel_pos = i - upstream
                    hits.append((rel_pos, score))
        else:
            for i in range(N - L + 1):
                # elementwise multiply (L×4) * (L×4) and sum → one float
                score = float(np.sum(S[i : i + L] * self.log_odds_revcomp))
                if score >= self.threshold:
                    rel_pos = i - upstream
                    hits.append((rel_pos, score))
        return hits
def get_motif_list(motif_inputs):
    motif_list: list[Motif] = []
    for entry in motif_inputs:
        typ   = entry["type"]
        data  = entry["data"]
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
        else:
            raise ValueError(f"Unknown motif type {typ}")
        m = Motif(name, color, pwm)
        motif_list.append(m)

    if not motif_list:
        raise ValueError("No valid motifs provided")
    return motif_list
#part 4: scan for motif hits

def scan_peaks_for_motifs(
    genome: Fasta,
    peaks_df: pd.DataFrame,
    window: int,
    motif_list: List[Motif],
    min_score: float
) -> Dict[str, Dict[str, List[Tuple[str,int,int,int,float]]]]:
    """
    Returns:
      motif_hits[motif_name][peak_id] = [
         (chrom, start, end, position, score),
         …
      ]
    """
    raw_hits: Dict[str, Dict[str, List[Tuple[str,int,int,int,float]]]] = {
        m.name: defaultdict(list) for m in motif_list
    }
    # faster than iterrows
    for chrom, midpt, peak_id, peak_len, strand in peaks_df[
        ['Chromosome','Midpoint','Peak_ID','Peak Length','Strand']
    ].itertuples(index=False):
        half = min(window, peak_len // 2)
        seq = genome[chrom][midpt-half: midpt+half].seq
        if strand == '-':
            seq = reverse_complement(seq)
        S = one_hot_encode(seq)
        for m in motif_list:
            # scan forward and reverse
            for hits in (m.scan_array(S, half, False), m.scan_array(S, half, True)):
                for pos, score in hits:
                    if score < min_score:
                        continue
                    start = midpt - half + pos
                    end = start + m.length
                    raw_hits[m.name][peak_id].append((chrom, start, end, pos, score))
    return raw_hits


def _strip_chr_name(name: str) -> str:
    return name[3:] if name.lower().startswith("chr") else name

def _strip_chr_iv(iv):
    """pybedtools.each() helper – modify iv.chrom in-place."""
    iv.chrom = _strip_chr_name(iv.chrom)
    return iv

def filter_motif_hits(
    motif_hits: Dict[str, Dict[str, List[Tuple[str, int, int, int, float]]]],
    atac_bed_fp: str,
) -> Dict[str, Dict[str, List[Tuple[str, int, int, int, float]]]]:
    # 1) flatten
    rows = [
        (motif, chrom, start, end, peak_id, dist, score)
        for motif, by_peak in motif_hits.items()
        for peak_id, hits in by_peak.items()
        for chrom, start, end, dist, score in hits
    ]
    if not rows:
        return {}

    motif_bed = (
        BedTool("\n".join("\t".join(map(str, r[1:])) for r in rows) + "\n",
                from_string=True)
        .each(_strip_chr_iv)
        .saveas())
    atac_bed = (
        BedTool(atac_bed_fp)
        .each(_strip_chr_iv)
        .saveas())
    print(f"[debug] motif hits loaded: {motif_bed.count()}")
    print(f"[debug] ATAC peaks loaded: {atac_bed.count()}")
    intersected = motif_bed.intersect(atac_bed, u=True)
    print(f"[debug] overlaps found: {intersected.count()}")
    motif_bed.saveas("motif_bed.bed")
    atac_bed.saveas("atac_bed.bed")

    #map back -> nested dict
    key_to_motif = {tuple(map(str, r[1:])): r[0] for r in rows}
    filtered: Dict[str, Dict[str, List[Tuple[str, int, int, int, float]]]] = {}
    for iv in intersected:
        chrom, start, end, peak_id, dist, score = iv.fields
        motif = key_to_motif.get((chrom, start, end, peak_id, dist, score))
        if motif is None:
            continue
        filtered.setdefault(motif, {}).setdefault(peak_id, []).append(
            (chrom, int(start), int(end), int(dist), float(score))
        )
    return filtered


def generate_plot(
    peaks_df: pd.DataFrame,
    motif_list: List[Motif],
    motif_hits: Dict[str, Dict[str, List[Tuple[str,int,int,int,float]]]],
    window_size: int,
    output_path: str = "motif_hits.png"
) -> Tuple[str, List[str], float]:
    """
    Just takes your (possibly filtered) motif_hits, plots, and returns:
      (output_path, peak_list, max_score)
    """
    # 1) Build peak_list in the same order as peaks_df
    peaks_with_hits = {
        peak_id
        for by_peak in motif_hits.values()
        for peak_id, hits in by_peak.items()
        if hits
    }
    print(peaks_with_hits)
    #if 'Strand' not in peaks_df.columns:
    peak_list = [pid for pid in peaks_df['Peak_ID'] if pid in peaks_with_hits]
   # else:
        #peak_list = list(peaks_df['Peak_ID'])

    # 2) Plot
    fig = plot_occurence_overview(peak_list, peaks_df, motif_list, motif_hits, window_size)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    # 3) Max score
    max_score = compute_max_motif_score(motif_hits)
    return output_path, peak_list, max_score


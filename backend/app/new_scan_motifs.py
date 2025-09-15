import numpy as np
import pandas as pd
import os, subprocess, tempfile, textwrap
from typing import Optional, List, Dict, Tuple
from pyfaidx import Fasta
from app.new_process_input import Motif

FIMO_PATH = "/home/ec2-user/miniconda3/envs/memesuite/bin/fimo"
#write the peaks_df to a meme-formatted txt file
def peaks_df_to_fasta(
    peaks_df: pd.DataFrame,
    reference_fasta: str,
    output_fasta: str,
    window_size: int,
):
    genome = Fasta(reference_fasta)

    with open(output_fasta, "w") as out:
        for chrom, midpt, peak_id, peak_len, strand in peaks_df[
        ['Chromosome','Midpoint','Peak_ID','Peak Length','Strand']].itertuples(index=False):
            half = min(window_size, peak_len // 2)
            zero_based_mid = midpt - 1
            start = max(0, zero_based_mid - half)
            end   = zero_based_mid + half + 1   # so length = 2*half + 1
            seq = genome[chrom][start : end].seq
            header = f">{peak_id}"
            out.write(header + "\n")
            # wrap at 60 chars per line
            for line in textwrap.wrap(seq, 60):
                out.write(line + "\n")


#write motifs to a MEME-formatted text file
def motif_to_memefile(motifs: List[Motif], filepath: str):

    if not motifs:
        raise ValueError("motif_to_memefile: received empty motif list")

    # Static header (up through the reference section)
    header = """\
********************************************************************************
STREME - Sensitive, Thorough, Rapid, Enriched Motif Elicitation
********************************************************************************
MEME version 5.5.6 (Release date: Wed Jun 19 13:59:04 2024 -0700)


********************************************************************************


********************************************************************************
REFERENCE
********************************************************************************
If you use this program in your research, please cite:

Timothy L. Bailey,
"STREME: accurate and versatile sequence motif discovery",
Bioinformatics, Mar. 24, 2021.
********************************************************************************

"""
    # The alphabet and strand lines
    header += """\
ALPHABET= ACGT

strands: + -

Background letter frequencies
"""

    #compute background
    bg = motifs[0].background
    bg_line = f"A {bg[0]:.6f} C {bg[1]:.6f} G {bg[2]:.6f} T {bg[3]:.6f}\n\n"

    with open(filepath, 'w') as out:
        out.write(header)
        out.write(bg_line)

        for idx, m in enumerate(motifs, start=1):
            L = m.pwm.shape[0]
            probs = m.pwm / m.pwm.sum(axis=1, keepdims=True)
            out.write(f"MOTIF {idx}-{m.name}\n")
            # we set nsites=0 and E=0 as placeholders
            out.write(f"letter-probability matrix: alength= 4 w= {L} nsites= 20 E= 0\n")
            for row in probs:
                out.write(" ".join(f"{p:.6f}" for p in row) + "\n")
            out.write("\n")


def run_fimo(
    meme_file: str, #path to the motif meme-formatted file
    sequences_file: str, #path to the fasta promoter sequence file
    output_dir: Optional[str] = None,
    threshold: float = 1e-4 #threshold for FIMO p-value for reporting hits.
) -> pd.DataFrame: 

    #Returns: fimo.tsv (motif_id, motif_alt_id, sequence_name, start, stop, strand, score, p-value, q-value, matched_sequence).
    with tempfile.TemporaryDirectory() as tmpdir:
        if output_dir is None:
            fimo_out = os.path.join(tmpdir, "fimo_out")
        else:
            fimo_out = output_dir
            os.makedirs(fimo_out, exist_ok=True)
        #run FIMO
        cmd = [
            "fimo",
            "--thresh", str(threshold),
            "--oc", fimo_out,
            meme_file,
            sequences_file
        ]
        subprocess.run(cmd, check=True)
        results_tsv = os.path.join(fimo_out, "fimo.tsv")
        try:
            df = pd.read_csv(results_tsv, sep="\t", comment="#")
        except pd.errors.EmptyDataError:
            print(f"No FIMO hits found at threshold {threshold}")
            df = pd.DataFrame()  # return empty dataframe

    return df



def build_motif_hits(
    fimo_df,
    peaks_df,
    motif_list
) -> Tuple[Dict[str, Dict[str, List[Tuple]]], pd.DataFrame]:
    
    peaks = peaks_df.set_index('Peak_ID')
    motif_hits = {m.name:{} for m in motif_list}
    for _, hit in fimo_df.iterrows():
        raw_mid   = hit['motif_id']
        mid = raw_mid.split('-', 1)[1] if '-' in raw_mid else raw_mid
        seqnm = hit['sequence_name']
        start = int(hit['start'])
        stop  = int(hit['stop'])
        strand= hit['strand']
        score = float(hit['score'])
        seq = str(hit['matched_sequence'])
        pval = float(hit['p-value'])

        # fetch the matching peak metadata
        if seqnm not in peaks.index:
            continue  # Skip hits without matching peak
        peak = peaks.loc[seqnm]
        peak_lfc = peak['logFC']
        peak_len = peak['Peak Length']
        chrom     = peak['Chromosome']
        seq0      = int(peak['Start'])      # genome‐coord of base 1 in your sequence
        # absolute genomic coords of the motif‐hit:
        abs_start = seq0 + (start - 1)
        abs_end   = seq0 + (stop  - 1)
        # relative to the TSS
        rel_pos   = int(start - peak_len//2)

        # stash the tuple in our nested dict
        motif_hits\
            .setdefault(mid, {})\
            .setdefault(seqnm, [])\
            .append((chrom,
                     abs_start,
                     abs_end,
                     rel_pos,
                     score,
                     strand,
                     peak_lfc, 
                     seq,
                     pval))
    rows = []
    for motif_name, by_gene in motif_hits.items():
        for gene, hits in by_gene.items():
            for chrom, start, end, rel_pos, score, strand, peak_lfc, seq, pval in hits:
                rows.append({
                    "Peak_ID":       gene,
                    "Motif":      motif_name,
                    "Chromosome":      chrom,
                    "Hit_start":      start,
                    "Hit_end":        end,
                    "Rel_pos":    rel_pos,
                    "Score_bits": score,
                    "Strand": strand,
                    "logFC": peak_lfc,
                    "Sequence": seq,
                    "p_value": pval,
                })
    hits_df = pd.DataFrame(rows)
    return motif_hits, hits_df

#wrapper function
def scan_wrapper(peaks_df, ref_fasta, window_size, motif_list:List[Motif], fimo_threshold) -> Dict[str, Dict[str, List[Tuple]]]:
    os.makedirs('fimo_files', exist_ok=True)
    fasta_fp = "fimo_files/sequences.txt"
    motifs_fp = "fimo_files/motifs.txt"
    peaks_df_to_fasta(peaks_df, ref_fasta, fasta_fp, window_size)
    motif_to_memefile(motif_list, motifs_fp)
    #fimo_tsv = run_fimo(motifs_fp, fasta_fp, "fimo_files", fimo_threshold)
    fimo_tsv = run_fimo(motifs_fp, fasta_fp, "fimo_files", fimo_threshold, fimo_exe=FIMO_PATH)
    motif_hits, df_hits = build_motif_hits(fimo_tsv, peaks_df, motif_list)
    return motif_hits, df_hits

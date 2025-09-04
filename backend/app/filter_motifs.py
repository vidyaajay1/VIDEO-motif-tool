import pybedtools
import pandas as pd


def _strip_chr_name(name: str) -> str:
    return name[3:] if name.lower().startswith("chr") else name

def _strip_chr_iv(iv):
   #pybedtools.each() helper to modify iv.chrom in-place.
    iv.chrom = _strip_chr_name(iv.chrom)
    return iv


def filter_motif_hits(hits_df: pd.DataFrame, bed_fp: str) -> pd.DataFrame:
    """
    Filters motif hits to retain only those overlapping regions in a BED file.

    Parameters
    ----------
    hits_df : pd.DataFrame
        DataFrame with columns in the order:
        ['peak_id', 'motif', 'chr', 'hit_start', 'hit_end', 'rel_pos', 'score', 'strand']
    bed_fp : str
        Path to BED file (e.g., ChIP or ATAC regions)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with the same columns and order as hits_df
    """
    # Build a BED-formatted string from hits_df
    bed_lines = "\n".join(
        f"{row.Chromosome}\t{row.Hit_start}\t{row.Hit_end}\t{row.Peak_ID}\t{row.Rel_pos}\t{row.Score_bits}\t{row.Strand}\t{row.logFC}\t{row.Sequence}\t{row.p_value}"
        for row in hits_df.itertuples(index=False)
    ) + "\n"


    motif_bed = pybedtools.BedTool(bed_lines, from_string=True).each(_strip_chr_iv).saveas()
    filter_bed = pybedtools.BedTool(bed_fp).each(_strip_chr_iv).saveas()

    print(f"[debug] motif hits loaded: {motif_bed.count()}")
    print(f"[debug] BED filter regions loaded: {filter_bed.count()}")

    intersected = motif_bed.intersect(filter_bed, u=True, f=0.5)
    print(f"[debug] overlaps found: {intersected.count()}")
    print("hits_df:\n", hits_df.head())
    # Extract filtered intervals
    filtered_records = []
    for iv in intersected:
        chrom, start, end, peak_id, rel_pos, score, strand, logFC, seq, pval = iv.fields
        filtered_records.append({
            "Chromosome": chrom,
            "Hit_start": int(start),
            "Hit_end": int(end),
            "Peak_ID": peak_id,
            "Rel_pos": int(rel_pos),
            "Score_bits": float(score),
            "Strand": strand,
            "logFC": float(logFC),
            "Sequence": str(seq),
            "p_value": float(pval)
        })

    # Convert to DataFrame
    filtered_df = pd.DataFrame(filtered_records)
    print(filtered_df.head())
    # Merge with original hits_df to recover motif and strand
    result = (
        filtered_df
        .merge(
            hits_df,
            on=["Chromosome", "Hit_start", "Hit_end", "Peak_ID", "Rel_pos", "Score_bits", "Strand", "logFC", "Sequence", "p_value"],
            how="left"
        )[["Peak_ID", "Motif", "Chromosome", "Hit_start", "Hit_end", "Rel_pos", "Score_bits", "Strand", "logFC", "Sequence", "p_value"]]
    )

    return result



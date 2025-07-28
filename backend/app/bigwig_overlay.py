import pyBigWig 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from app.process_motifs import Motif

#part 6: overlay chip/atac track plot
def fetch_bw_coverage(peak_id: str, peaks_df: pd.DataFrame, bw_input: list[tuple[str,str,str]]) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    #bw_input is a list of [(bigwig_filepath, track_name, track_color)]
    coverage_data = []
    if "Chromosome_chr" not in peaks_df.columns:
        peaks_df = peaks_df.copy()
        peaks_df["Chromosome_chr"] = "chr" + peaks_df["Chromosome"].astype(str)
    row = peaks_df[peaks_df['Peak_ID'] == peak_id].iloc[0]
    midpt = row["Midpoint"]
    start, end = row["Start"], row["End"]
    for path, label, color in bw_input:
        try:
            bw = pyBigWig.open(path)
        except:
            continue
        header = bw.chroms() #dict of { "chr2L": length, "chr3R": length, ... } or { "2L": length, ... }
        if row["Chromosome"] in header: #the bigwig uses 2L, 3R, etc
            used_chrom = row["Chromosome"]
        elif row["Chromosome_chr"] in header:
            used_chrom = row["Chromosome_chr"]
        else:
            print(f"Skipped {label}: chromosome not found or invalid region.")
            bw.close(); continue #skip if neither is found
        
        chrom_len = header[used_chrom]
        s = max(0, start)
        e = min(end, chrom_len)
        if e <= s:
            bw.close()
            continue
        
        raw_coverage = bw.values(used_chrom, s, e)
        vals = np.nan_to_num(raw_coverage)
        pos  = np.arange(s, e) - midpt
        coverage_data.append((label, pos, vals, color))
        bw.close()
    return coverage_data

def plot_chip_overlay(
    coverage_data: list[tuple[str, np.ndarray, np.ndarray, str]],
    motifs: list[Motif], 
    motif_hits: dict[str, dict[str, list[tuple[int,float]]]],
    peak_id: str,
    window: int,
    output_path: str
) -> plt.Figure:
    
    fig, axes = plt.subplots(
        len(coverage_data) + 1, 1,
        sharex=True,
        figsize=(8, (len(coverage_data) + 1) * 0.5),
        gridspec_kw={'height_ratios': [0.5] + [1] * len(coverage_data), 'hspace': 0.1},
        constrained_layout = True
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    #plot motifs above tracks
    ax0 = axes[0]
    for m in motifs:
        for _, _, _, pos, score, _, _ in motif_hits[m.name].get(peak_id, []):
            raw_alpha = 0.7 * (score / m.max_score) + 0.3
            alpha = max(0.0, min(1.0, raw_alpha))
            ax0.broken_barh([(pos, m.length)], (0,1),
                            facecolors=m.color, edgecolor='black',
                            lw=0.3, alpha=alpha)
    ax0.axis('off')
    #plot tracks
    gmax = max((vals.max() for _, _, vals, _ in coverage_data if vals.size), default=1.0)
    for i, (label, pos, vals, col) in enumerate(coverage_data, start=1):
        ax = axes[i]
        if pos.size:
            ax.plot(pos, vals, color=col, lw=0.6)
            ax.fill_between(pos, vals, 0, color=col, alpha=0.3)
        ax.set_xlim(-window, window)
        ax.set_ylim(0, gmax)
        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8)
        ax.tick_params(axis='y', labelsize=6)
        ax.spines['top'].set_visible(False)

    axes[-1].set_xlabel(f'±{window} bp from Peak Midpoint/ Gene TSS')
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig
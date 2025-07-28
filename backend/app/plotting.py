import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import re
from typing import Optional

def rank_peaks_for_plot(
    df_hits: pd.DataFrame,
    gene_lfc: dict[str, float],
    peaks_df: pd.DataFrame,
    use_hit_number: bool,
    use_match_score: bool,
    motif: Optional[str],
) -> dict[str, int]:
    """
    Returns a dict mapping Peak_ID → rank (1 = highest).
    In the 'real' case it ranks each peak by hit‐number or score.
    In the fallback (motif=None or no flags), it ranks peaks
    by the gene‐level dummy LFC, grouping peaks by gene.
    """
    # helper to extract gene
    gene_from_peak = peaks_df["Peak_ID"].str.split("_FBtr").str[0]

    # 1) FALLBACK: no motif or flags
    if motif is None or not (use_hit_number or use_match_score):
        # build a DataFrame of peaks + gene + dummy metric
        fb = pd.DataFrame({
            "Peak_ID": peaks_df["Peak_ID"],
            "gene": gene_from_peak,
        })
        fb["metric"] = fb["gene"].map(lambda g: gene_lfc.get(g, 0.0))
        # stable sort by metric desc (ties keep original order)
        fb_sorted = fb.sort_values("metric", ascending=False, kind="mergesort")

    else:
        # 2) REAL ranking: only hits for this motif
        df_m = df_hits[df_hits["Motif"] == motif]
        if df_m.empty:
            # fallback into same logic as above
            return rank_peaks_for_plot(
                df_hits, gene_lfc, peaks_df,
                use_hit_number=False, use_match_score=False, motif=None
            )

        # build the aggregation
        agg: dict[str, tuple] = {}
        if use_hit_number:
            agg["hit_count"] = ("Peak_ID", "size")
        if use_match_score:
            agg["metric"] = (
                "Score_bits",
                "sum" if use_hit_number else "max"
            )

        metrics = df_m.groupby("Peak_ID").agg(**agg)
        if "metric" not in metrics:
            # only hit_count was requested
            metrics["metric"] = metrics["hit_count"]

        # merge back onto peaks, fill non‐hits with 0
        fb = pd.DataFrame({"Peak_ID": peaks_df["Peak_ID"]})
        fb = fb.merge(
            metrics[["metric"]].reset_index(),
            on="Peak_ID", how="left"
        ).fillna(0.0)
        fb["gene"] = gene_from_peak
        # tie‐breaker: gene_lfc
        fb["fallback"] = fb["gene"].map(lambda g: abs(gene_lfc.get(g, 0.0)))

        # sort by (metric desc, fallback desc)
        fb_sorted = fb.sort_values(
            by=["metric", "fallback"],
            ascending=[False, False],
            kind="mergesort"
        )

    # 3) Assign peak ranks
    peak_rank = {
        pid: rank + 1
        for rank, pid in enumerate(fb_sorted["Peak_ID"])
    }
    return peak_rank

def plot_occurence_overview(
    peak_list: list[str],
    peaks_df: pd.DataFrame,
    motifs: list,
    motif_hits: dict[str, dict[str, list]] = None,
    df_hits: pd.DataFrame = None,
    window: int = 1000,
    peak_rank: dict[str, int] = None,     # renamed from gene_rank
    output_path: str = "motif_hits.png"
):
    # 1) apply ranking
    if peak_rank:
        peak_list = sorted(peak_list, key=lambda pid: peak_rank.get(pid, float("inf")))

    # 2) if the user actually asked for a filtered plot (df_hits not None) but got zero hits:
    if df_hits is not None and df_hits.empty:
        fig, ax = plt.subplots(figsize=(8,2))
        ax.text(.5, .5, "No motif hits found…", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=300); plt.close(fig)
        return output_path, peak_list

    # 3) plotting loop is unchanged
    fig, ax = plt.subplots(figsize=(7, max(len(peak_list)*0.3, 2)))
    for row_idx, peak in enumerate(peak_list):
        row = peaks_df.loc[peaks_df["Peak_ID"] == peak].iloc[0]
        rel_start, rel_end = row.Start - row.Midpoint, row.End - row.Midpoint

        ax.broken_barh(
            [(rel_start, rel_end-rel_start)],
            (row_idx - 0.3, 0.6),
            facecolors='lightgray', lw=1.0, linestyle='dashed'
        )


        if df_hits is not None:
            df_p = df_hits[df_hits["Peak_ID"] == peak]
            for m in motifs:
                for _, hit in df_p[df_p["Motif"] == m.name].iterrows():
                    alpha = max(0, min(1, 0.7*(hit.Score_bits/m.max_score)+0.3))
                    ax.broken_barh(
                        [(hit.Rel_pos, m.length)],
                        (row_idx - 0.4, 0.8),
                        facecolors=m.color,
                        edgecolor='black',
                        lw=0.2,
                        alpha=alpha
                    )

        ax.plot([-window, window], [row_idx+0.5]*2, color='white', lw=0.8)

    # 4) finish styling
    handles = [Patch(facecolor=m.color, edgecolor='black', label=m.name) for m in motifs]
    ax.legend(handles=handles, title='Motifs', loc='upper right',
              fontsize=6, title_fontsize=8, frameon=False)
    ax.set_xlim(-window, window)
    ax.set_ylim(-0.5, len(peak_list)-0.5)
    ax.set_yticks(range(len(peak_list)))
    ax.set_yticklabels(peak_list, fontsize=6)
    ax.set_facecolor('lightgray')
    ax.axvline(0, color='white', lw=1)
    ax.invert_yaxis()
    ax.set_xlabel(f'±{window} bp from Peak Midpoint')
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path, peak_list

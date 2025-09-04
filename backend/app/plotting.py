from matplotlib.colors import to_rgba
import pandas as pd
import numpy as np
import re
from typing import Optional, List, Dict
from dataclasses import dataclass
import plotly.graph_objects as go


@dataclass
class MotifSpec:
    name: str
    length: int
    color: str
    max_score: float
    #ADD A P-VAL THRESHOLD?
def _as_motif_specs(motif_list, df_hits: pd.DataFrame) -> List[MotifSpec]:
    specs: List[MotifSpec] = []
    by_motif_max = df_hits.groupby("Motif")["Score_bits"].max().to_dict()

    for m in motif_list:
        # Access like object OR dict
        name   = getattr(m, "name", None) or m.get("name")
        color  = getattr(m, "color", None) or m.get("color", "#1f77b4")
        length = getattr(m, "length", None) or m.get("length")
        max_s  = getattr(m, "max_score", None) or m.get("max_score")
        #UPDATE THRESHOLD BASED ON SLIDER (FRONTEND INPUT)?
        if max_s is None:
            max_s = by_motif_max.get(name, 1.0) or 1.0
        if length is None:
            # Try to infer from a column 'length' in hits (if you stored it) else default 8
            length = int(df_hits.loc[df_hits["Motif"] == name, "length"].iloc[0]) if (
                "length" in df_hits.columns and (df_hits["Motif"] == name).any()
            ) else 8
        specs.append(MotifSpec(name=name, length=int(length), color=str(color), max_score=float(max_s)))
    return specs

def filter_hits_by_per_motif_pvals(
    df_hits: pd.DataFrame,
    per_motif_pvals: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Keep rows whose motif-specific p-value <= threshold for that motif.
    If a motif has no threshold, keep all its hits.
    Assumes df_hits has a 'p-value' column.
    """
    if not isinstance(df_hits, pd.DataFrame) or df_hits.empty:
        return df_hits
    if not per_motif_pvals:
        return df_hits

    thr = df_hits["Motif"].map(per_motif_pvals)
    mask = thr.isna() | (df_hits["p_value"].astype(float) <= thr.astype(float))
    return df_hits.loc[mask].copy()


def rank_peaks_for_plot(
    df_hits: pd.DataFrame,
    gene_lfc: Dict[str, float],
    peaks_df: pd.DataFrame,
    use_hit_number: bool,
    use_match_score: bool,
    motif: Optional[str],
    min_score_bits: float = 0.0,
    per_motif_pvals: Optional[Dict[str, float]] = None,   # NEW
) -> Dict[str, int]:
    gene_from_peak = peaks_df["Peak_ID"].str.split("_FBtr").str[0]

    # NEW: per-motif p-value filter first
    df_hits = filter_hits_by_per_motif_pvals(df_hits, per_motif_pvals)

    if motif is None or not (use_hit_number or use_match_score):
        fb = pd.DataFrame({"Peak_ID": peaks_df["Peak_ID"], "gene": gene_from_peak})
        fb["metric"] = fb["gene"].map(lambda g: gene_lfc.get(g, 0.0)).clip(lower=0.0)
        fb_sorted = fb.sort_values("metric", ascending=False, kind="mergesort")
    else:
        df_m = df_hits[df_hits["Motif"] == motif].copy()
        df_m = df_m[df_m["Score_bits"] > min_score_bits]  # keep your bits gate

        if df_m.empty:
            return rank_peaks_for_plot(
                df_hits, gene_lfc, peaks_df,
                use_hit_number=False, use_match_score=False, motif=None
            )

        agg = {}
        if use_hit_number:
            agg["hit_count"] = ("Peak_ID", "size")
        if use_match_score:
            agg["metric"] = ("Score_bits", "sum" if use_hit_number else "max")

        metrics = df_m.groupby("Peak_ID").agg(**agg)
        if "metric" not in metrics:
            metrics["metric"] = metrics["hit_count"]

        fb = pd.DataFrame({"Peak_ID": peaks_df["Peak_ID"]})
        fb = fb.merge(metrics[["metric"]].reset_index(), on="Peak_ID", how="left").fillna(0.0)
        fb["metric"] = fb["metric"].clip(lower=0.0)
        fb["fallback"] = gene_from_peak.map(lambda g: abs(gene_lfc.get(g, 0.0)))
        fb_sorted = fb.sort_values(by=["metric", "fallback"], ascending=[False, False], kind="mergesort")

    return {pid: i+1 for i, pid in enumerate(fb_sorted["Peak_ID"])}


# --- 4) Plotly figure builder -------------------------------------------------
def _rgba_with_alpha(col: str, alpha: float) -> str:
    r, g, b, _ = to_rgba(col)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{max(0.0, min(1.0, alpha))})"


def plot_occurrence_overview_plotly(
    peak_list: List[str],
    peaks_df: pd.DataFrame,
    motif_list: list,
    df_hits: Optional[pd.DataFrame] = None,
    window: int = 500,
    peak_rank: Optional[Dict[str, int]] = None,
    title: str = "Motif positions in genomic regions",
    min_score_bits: float = 0.0,
    per_motif_pvals: Optional[Dict[str, float]] = None,   # NEW
):
    if peak_rank:
        peak_list = sorted(peak_list, key=lambda pid: peak_rank.get(pid, float("inf")))
    peak_list = [str(p) for p in peak_list]

    motifs = _as_motif_specs(motif_list, df_hits)

    sub_all = None
    if df_hits is not None and len(df_hits):
        # NEW: per-motif p-value filter, then (optional) bits filter
        hits = filter_hits_by_per_motif_pvals(df_hits, per_motif_pvals)
        if min_score_bits is not None:
            hits = hits[hits["Score_bits"] > float(min_score_bits)]
        hits["Peak_ID"] = hits["Peak_ID"].astype(str)
        sub_all = hits[hits["Peak_ID"].isin(peak_list)]


    # Build background bars for ALL peaks (even if there are no hits)
    dfp = peaks_df.copy()
    dfp["Peak_ID"] = dfp["Peak_ID"].astype(str)

    bg_rows = []
    for peak in peak_list:
        row = dfp.loc[dfp["Peak_ID"] == peak].iloc[0]
        rel_start = int(row.Start - row.Midpoint)
        rel_end   = int(row.End   - row.Midpoint)
        bg_rows.append({"Peak_ID": peak, "base": rel_start, "length": rel_end - rel_start})
    df_bg = pd.DataFrame(bg_rows)

    fig = go.Figure()

    # Background (no offsetgroup; keep alignmentgroup consistent)
    fig.add_bar(
        name="Peak window",
        y=df_bg["Peak_ID"], x=df_bg["length"], base=df_bg["base"], orientation="h",
        marker=dict(color="rgba(200,200,200,0.35)", line=dict(color="rgba(90,90,90,0.7)", width=0.5)),
        hoverinfo="skip", showlegend=False,
        alignmentgroup="peaks", width=0.8,
    )

    # Legend header
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=0, color="rgba(0,0,0,0)"),
                             showlegend=True, name="<b>Motifs</b>", hoverinfo="skip"))

    # Motif bars (only for peaks that *have* hits)
    if sub_all is not None and not sub_all.empty:
        for m in _as_motif_specs(motif_list, df_hits):
            sub = sub_all[sub_all["Motif"] == m.name]
            if sub.empty:
                continue

            x_vals    = np.full(len(sub), m.length, dtype=float)
            base_vals = sub["Rel_pos"].astype(float).values
            y_vals    = sub["Peak_ID"].astype(str).values

            chrom     = sub["Chromosome"].astype(str).values
            start_abs = sub["Hit_start"].astype(int).values
            end_abs   = sub["Hit_end"].astype(int).values
            strand    = sub["Strand"].astype(str).values
            pvals     = sub["p_value"].astype(float).values   # NEW

            denom  = m.max_score if m.max_score else 1.0
            alphas = 0.3 + 0.7 * np.clip(sub["Score_bits"].astype(float).values / denom, 0, 1)
            colors = [_rgba_with_alpha(m.color, a) for a in alphas]

            end_vals = base_vals + x_vals
            seq = sub["Sequence"] if "Sequence" in sub.columns else None

            # NEW: include p-value in hover
            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "chr%{customdata[4]}: %{customdata[5]:,} - %{customdata[6]:,}<br>"
                "Score: %{customdata[3]:.2f}<br>"
                "p-value: %{customdata[9]:.2e}<br>"   # index updated below
                "Strand: %{customdata[7]}"
                + ("<br>Seq: %{customdata[8]}" if seq is not None else "")
                + "<extra></extra>"
            )

            custom_cols = [
                np.full(len(sub), m.name),
                base_vals, end_vals,
                sub["Score_bits"].astype(float).values,
                chrom, start_abs, end_abs, strand,
            ]
            if seq is not None:
                custom_cols.append(seq.values)  # becomes index 8
            # Append pvals as the last column (index 9 if seq present, else 8)
            custom_cols.append(pvals)
            custom = np.column_stack(custom_cols)

            fig.add_bar(
                name=m.name,
                y=y_vals, x=x_vals, base=base_vals, orientation="h",
                marker=dict(color=colors, line=dict(color="black", width=0.4)),
                hovertemplate=hovertemplate, customdata=custom,
                alignmentgroup="peaks", width=0.8,
            )
    else:
        fig.add_annotation(text="No motif hits pass the per-motif p-value filter.",
                           showarrow=False, x=0.5, y=1.02, xref="paper", yref="paper")

    # Midline and layout
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(peak_list)-0.5,
                  line=dict(color="rgba(80,80,80,0.6)", width=1), layer="above")

    fig.update_layout(
        title=title,
        template="plotly_white",
        barmode="overlay",
        bargap=0.15, bargroupgap=0.0,
        margin=dict(l=160, r=12, t=60, b=40),
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(255,255,255,0.6)", borderwidth=0),
        height=max(240, int(len(peak_list) * 30)),
        autosize=True,
        xaxis=dict(range=[-window, window], constrain="domain", zeroline=True,
                   title="Position relative to TSS/peak center"),
    )
    fig.update_yaxes(
        type="category",
        categoryorder="array",
        categoryarray=peak_list,   # all peaks, including no-hit ones
        autorange="reversed",
    )

    return fig, peak_list




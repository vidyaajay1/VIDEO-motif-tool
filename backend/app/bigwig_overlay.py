# bigwigoverlay.py  (Plotly version)

import pyBigWig
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# pull helpers from your new plotting module
from app.plotting import (
    _as_motif_specs,
    _rgba_with_alpha,
    filter_hits_by_per_motif_pvals,
)

# --------------------------------------------------------------------
# part 6: overlay chip/atac track plot
# --------------------------------------------------------------------
def fetch_bw_coverage(
    peak_id: str,
    peaks_df: pd.DataFrame,
    bw_input: List[Tuple[str, str, str]],
) -> List[Tuple[str, np.ndarray, np.ndarray, str]]:
    """
    bw_input: list of tuples (bigwig_filepath, track_name, track_color)
    returns:  list of tuples (label, rel_positions, values, color)
    """
    coverage_data: List[Tuple[str, np.ndarray, np.ndarray, str]] = []

    if "Chromosome_chr" not in peaks_df.columns:
        peaks_df = peaks_df.copy()
        peaks_df["Chromosome_chr"] = "chr" + peaks_df["Chromosome"].astype(str)

    row = peaks_df[peaks_df["Peak_ID"] == peak_id].iloc[0]
    midpt = int(row["Midpoint"])
    start, end = int(row["Start"]), int(row["End"])

    for path, label, color in bw_input:
        try:
            bw = pyBigWig.open(path)
        except Exception:
            # skip unreadable files
            continue

        header = bw.chroms()  # dict like {"chr2L": length, "chr3R": length} or {"2L": length}
        if str(row["Chromosome"]) in header:
            used_chrom = str(row["Chromosome"])
        elif str(row["Chromosome_chr"]) in header:
            used_chrom = str(row["Chromosome_chr"])
        else:
            print(f"Skipped {label}: chromosome not found or invalid region.")
            bw.close()
            continue

        chrom_len = int(header[used_chrom])
        s = max(0, start)
        e = min(end, chrom_len)
        if e <= s:
            bw.close()
            continue

        raw_coverage = bw.values(used_chrom, s, e)
        vals = np.nan_to_num(raw_coverage).astype(float)
        pos = np.arange(s, e, dtype=int) - midpt
        coverage_data.append((label, pos, vals, color))
        bw.close()

    return coverage_data


def plot_chip_overlay(
    *,
    coverage_data: List[Tuple[str, np.ndarray, np.ndarray, str]],
    motif_list: List,
    df_hits: pd.DataFrame,
    peaks_df: pd.DataFrame,
    peak_id: str,
    window: int = 500,
    per_motif_pvals: Optional[Dict[str, float]] = None,
    min_score_bits: float = 0.0,
    title: str = "Motif hits with ChIP/ATAC coverage",
    output_html: Optional[str] = None,
):

    # --- filter hits to this peak + p-value + bits threshold ---
    if df_hits is None or df_hits.empty:
        sub = pd.DataFrame(columns=["Motif"])
    else:
        sub = df_hits[df_hits["Peak_ID"].astype(str) == str(peak_id)].copy()
        sub = filter_hits_by_per_motif_pvals(sub, per_motif_pvals)
        if min_score_bits is not None:
            sub = sub[sub["Score_bits"].astype(float) > float(min_score_bits)]

    # motif specs (name, length, color, max_score)
    specs = _as_motif_specs(motif_list, df_hits if df_hits is not None else pd.DataFrame())

    # layout rows: 1 for motifs + one per coverage track
    n_tracks = len(coverage_data)
    n_rows = n_tracks + 1
    # Make the motifs row short like before
    row_heights = [0.25] + [1.0] * n_tracks

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        specs=[[{"type": "bar"}]] + [[{"type": "xy"}] for _ in range(n_tracks)],
    )

    # -------- Row 1: Motif rectangles as horizontal bars ----------
    # background band representing the peak window
    # figure out relative start/end for this peak
    rowp = peaks_df[peaks_df["Peak_ID"].astype(str) == str(peak_id)].iloc[0]
    rel_start = int(rowp["Start"] - rowp["Midpoint"])
    rel_end = int(rowp["End"] - rowp["Midpoint"])
    fig.add_trace(
        go.Bar(
            x=[rel_end - rel_start],
            y=["motifs"],
            base=[rel_start],
            orientation="h",
            marker=dict(color="rgba(200,200,200,0.35)", line=dict(color="rgba(90,90,90,0.7)", width=0.5)),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # one bar per hit, colored by motif with alpha ~ score / max_score
    if not sub.empty:
        # show a "Motifs" legend header (empty scatter)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=0),
                showlegend=True,
                name="<b>Motifs</b>",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        for m in specs:
            sub_m = sub[sub["Motif"] == m.name]
            if sub_m.empty:
                continue

            x_vals = np.full(len(sub_m), m.length, dtype=float)
            base_vals = sub_m["Rel_pos"].astype(float).values
            denom = m.max_score if m.max_score else 1.0
            alphas = 0.3 + 0.7 * np.clip(sub_m["Score_bits"].astype(float).values / denom, 0, 1)
            colors = [_rgba_with_alpha(m.color, a) for a in alphas]

            chrom = sub_m.get("Chromosome", pd.Series(["?"] * len(sub_m))).astype(str).values
            start_abs = sub_m.get("Hit_start", pd.Series([np.nan] * len(sub_m))).astype("Int64").values
            end_abs = sub_m.get("Hit_end", pd.Series([np.nan] * len(sub_m))).astype("Int64").values
            strand = sub_m.get("Strand", pd.Series(["."] * len(sub_m))).astype(str).values
            pvals = sub_m.get("p_value", pd.Series([np.nan] * len(sub_m))).astype(float).values
            seq = sub_m["Sequence"].values if "Sequence" in sub_m.columns else None

            end_vals = base_vals + x_vals

            custom_cols = [
                np.full(len(sub_m), m.name),
                base_vals,
                end_vals,
                sub_m["Score_bits"].astype(float).values,
                chrom,
                start_abs,
                end_abs,
                strand,
            ]
            if seq is not None:
                custom_cols.append(seq)  # index 8
            custom_cols.append(pvals)   # last column (9 if seq exists else 8)
            custom = np.column_stack(custom_cols)

            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "chr%{customdata[4]}: %{customdata[5]:,} - %{customdata[6]:,}<br>"
                "Score: %{customdata[3]:.2f}<br>"
                "p-value: %{customdata[9]:.2e}<br>"
                "Strand: %{customdata[7]}"
                + ("<br>Seq: %{customdata[8]}" if seq is not None else "")
                + "<extra></extra>"
            )

            # We draw each hit as a horizontal bar of width=length at base=Rel_pos
            fig.add_trace(
                go.Bar(
                    name=m.name,
                    x=x_vals,
                    y=["motifs"] * len(x_vals),
                    base=base_vals,
                    orientation="h",
                    marker=dict(color=colors, line=dict(color="black", width=0.4)),
                    hovertemplate=hovertemplate,
                    customdata=custom,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
    else:
        fig.add_annotation(
            text="No motif hits pass filters.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",   # not "x1 domain"
            yref="y domain",   # not "y1 domain"
            xanchor="center",
            yanchor="middle",
            row=1,
            col=1,
        )

    # hide y labels for motifs row
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    # vertical midline at 0 spanning the full figure
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="rgba(80,80,80,0.6)", width=1),
        layer="above",
    )

    # -------- Rows 2..N: coverage tracks ----------
    # global ymax for consistent scaling 
    gmax = max((vals.max() for _, _, vals, _ in coverage_data if vals.size), default=1.0)

    for i, (label, pos, vals, col) in enumerate(coverage_data, start=2):
        if pos.size:
            # line
            fig.add_trace(
                go.Scatter(
                    x=pos,
                    y=vals,
                    mode="lines",
                    name=label,
                    line=dict(width=1.0, color=col),
                    showlegend=False,
                ),
                row=i,
                col=1,
            )
            # fill
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([pos, pos[::-1]]),
                y=np.concatenate([vals, np.zeros_like(vals)]),
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor=_rgba_with_alpha(col, 0.3),
                name=f"{label} (area)",
                showlegend=False,
                hoverinfo="skip",      # <- hide hover on the area
            ),
            row=i,
            col=1,
        )

        # y-axis styling
        fig.update_yaxes(
            title_text=label,
            title_standoff=0,
            rangemode="tozero",
            range=[0, float(gmax) if np.isfinite(gmax) else 1.0],
            ticks="outside",
            tickfont=dict(size=9),
            row=i,
            col=1,
        )
    # -------- Global layout ----------
    fig.update_xaxes(
        range=[-int(window), int(window)],
        zeroline=True,
        zerolinewidth=1,
    )
    fig.update_xaxes(
    title=f"±{window} bp from Peak Midpoint / Gene TSS",
    row=n_rows,  # bottom row (motifs is row 1)
    col=1,
)
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=80, r=16, t=60, b=40),
        height=max(320, int((n_rows) * 120)),
        barmode="overlay",
        bargap=0.15,
        bargroupgap=0.0,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="right", x=1.0, bgcolor="rgba(255,255,255,0.6)"),
    )

    if output_html:
        fig.write_html(output_html, include_plotlyjs="cdn")

    return fig

import numpy as np
import pandas as pd
from app.new_process_input import Motif

def raw_scoring_promoter(hits_df:pd.DataFrame):
    # 1) Promoter presence/number
    promo = (
        hits_df
        .groupby(["Peak_ID","Motif"])
        .size()                      # count hits
        .rename("N_prom")            # number of promoter hits
        .to_frame()
    )
    promo["M_prom"] = (promo["N_prom"] > 0).astype(int)
    return promo

def raw_scoring_chip(chip_hits: pd.DataFrame, df_hits: pd.DataFrame):
    """
    chip_hits: filtered DataFrame with only overlapping hits
    df_hits: original (unfiltered) DataFrame of all motif hits
    """
    # Step 1: Get all (Peak_ID, Motif) combinations from unfiltered data
    all_pairs = df_hits[["Peak_ID", "Motif"]].drop_duplicates()
    all_index = pd.MultiIndex.from_frame(all_pairs)

    # Step 2: Score based on filtered data
    chip = (
        chip_hits
        .groupby(["Peak_ID", "Motif"])
        .size()
        .rename("N_chip")
        .to_frame()
    )

    # Step 3: Add missing pairs as 0s
    chip = chip.reindex(all_index, fill_value=0)

    # Step 4: Add M_chip
    chip["M_chip"] = (chip["N_chip"] > 0).astype(int)
    print("chip_df\n", chip.head())
    return chip

def raw_scoring_atac(atac_hits: pd.DataFrame, df_hits: pd.DataFrame):

    all_pairs = df_hits[["Peak_ID", "Motif"]].drop_duplicates()
    all_index = pd.MultiIndex.from_frame(all_pairs)
    atac = (
        atac_hits
        .groupby(["Peak_ID", "Motif"])
        .size()
        .rename("N_atac")
        .to_frame()
    )
    atac = atac.reindex(all_index, fill_value=0)
    atac["M_atac"] = (atac["N_atac"] > 0).astype(int)
    return atac


#wrapper
def score_and_merge(hits_df, chip_hits_df, atac_hits_df):
    print("hits_df:\n", hits_df.head())

    promo = raw_scoring_promoter(hits_df)
    chip = raw_scoring_chip(chip_hits_df, hits_df)
    atac = raw_scoring_atac(atac_hits_df, hits_df)

    merged = promo.join(chip, how='outer').join(atac, how='outer')
    merged = merged.fillna(0)

    # Step 4: Optional: convert float columns to int
    merged = merged.astype({
        "N_prom": int,
        "M_prom": int,
        "N_chip": int,
        "M_chip": int,
        "N_atac": int,
        "M_atac": int,
    })
    if "logFC" in hits_df.columns:
        # Extract Peak_ID → logFC mapping (drop duplicates)
        lfc_df = hits_df[["Peak_ID", "logFC"]].drop_duplicates().set_index("Peak_ID")
        merged = merged.join(lfc_df, how='left')

    # Step 6: Add average FIMO Score_bits if available
    if "Score_bits" in hits_df.columns:
        # Group by Peak_ID and Motif to get average FIMO score
        score_df = (
            hits_df.groupby(["Peak_ID", "Motif"])["Score_bits"]
            .mean()
            .rename("FIMO_score")
        )
        # Join using multi-index (Peak_ID, Motif)
        merged = merged.join(score_df, how='left')

    return merged


def score_hit_naive_bayes(
    scored_df: pd.DataFrame,
    feature_likelihoods: dict = None,
    prior_direct: float = 0.05,
    pseudocount: float = 1e-6,
    return_all: bool = True
) -> pd.DataFrame:
    """
    Scores each motif hit using Naïve Bayes with dynamic logFC discretization.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: M_prom, M_chip, M_atac, logFC.
    feature_likelihoods : dict
        Format: {feature_name: {value: P(value | direct)}}
    prior_direct : float
        Prior probability that a motif hit is regulatory.
    pseudocount : float
        Used to prevent log(0) or divide-by-zero errors.
    return_all : bool
        If True, returns df with added 'P_regulatory' column.

    Returns
    -------
    pd.DataFrame or pd.Series
    """

    df = scored_df.copy()

    # Compute dynamic thresholds based on logFC quantiles
    valid_logfc = df["logFC"].dropna()
    if not valid_logfc.empty:
        q_low, q_high = valid_logfc.quantile([0.33, 0.66])
    else:
        q_low, q_high = -1.0, 1.0  # fallback

    def discretize_logfc(logfc):
        if pd.isna(logfc):
            return "neutral"
        elif logfc >= q_high:
            return "up"
        elif logfc <= q_low:
            return "down"
        else:
            return "neutral"

    df["logFC_cat"] = df["logFC"].apply(discretize_logfc)

    valid_fimo = df["FIMO_score"].dropna()
    if not valid_fimo.empty:
        fimo_q_low, fimo_q_high = valid_fimo.quantile([0.33, 0.66])
    else:
        fimo_q_low, fimo_q_high = 5.0, 15.0  # fallback

    def discretize_fimo_score(score):
        if pd.isna(score):
            return "med"
        elif score >= fimo_q_high:
            return "high"
        elif score <= fimo_q_low:
            return "low"
        else:
            return "med"

    df["FIMO_score_cat"] = df["FIMO_score"].apply(discretize_fimo_score)
    # Default likelihoods if not provided
    if feature_likelihoods is None:
        feature_likelihoods = {
            "M_prom": {1: 0.75, 0: 0.25},
            "M_chip": {1: 0.90, 0: 0.10},
            "M_atac": {1: 0.85, 0: 0.30},
            "logFC_cat": {
                "up": 0.80,
                "neutral": 0.30,
                "down": 0.05
            },
            "FIMO_score_cat": {
                "high": 0.85,
                "med": 0.5,
                "low": 0.2
            }
        }

    prior_nondirect = 1 - prior_direct

    def compute_posterior(row):
        num = prior_direct
        denom = prior_nondirect

        for feat, probs in feature_likelihoods.items():
            val = row.get(feat, None)
            if pd.isna(val):
                val = "neutral" if feat == "logFC_cat" else 0
            p = probs.get(val, pseudocount)
            num *= p
            denom *= 1 - p

        return num / (num + denom + pseudocount)

    scores = df.apply(compute_posterior, axis=1)
    scores.name = "P_regulatory"

    return df.assign(P_regulatory=scores) if return_all else scores
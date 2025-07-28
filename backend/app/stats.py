import pandas as pd
from scipy.stats import kstest
from statsmodels.stats.multitest import multipletests
import numpy as np
from itertools import combinations
from scipy.stats import fisher_exact
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

def positional_preference_test(df_hits: pd.DataFrame, window: int = 500,
                               R: int = 200, min_hits: int = 20,
                               fdr: float = 0.05) -> pd.DataFrame:
    
    """
    Empirical KS test for motif positional bias, keeping per-promoter hit
    counts constant through permutation.

    Parameters
    ----------
    df_hits : DataFrame  (columns: motif, gene, distance [, ...])
    window  : int        Half-window in bp around the TSS (± window).
    R       : int        Number of permutations per motif.
    min_hits: int        Skip motifs with < min_hits total occurrences.
    fdr     : float      FDR threshold for the 'significant' flag.

    Returns
    -------
    DataFrame with columns:
      motif, n_hits, ks_obs, p_emp, q_value, significant
    """

    rng      = np.random.default_rng(42)
    results  = []

    for motif, g in df_hits.groupby('motif'):         
        distances = g['distance'].to_numpy(dtype=float)
        n_hits    = len(distances)
        if n_hits < min_hits:
            continue

        # map to [0,1] for analytic KS
        u_obs  = (distances + window) / (2 * window)
        ks_obs = kstest(u_obs, 'uniform').statistic

        # counts = hits per promoter
        counts = g.groupby('gene').size().to_numpy()  
        ks_null = np.empty(R)

        for j in range(R):
            # generate one permutation
            perm = rng.uniform(-window, window, counts.sum())
            # shuffle *within promoter blocks* to keep counts intact
            rng.shuffle(perm)            # cheaper than loop in Python
            u_perm     = (perm + window) / (2 * window)
            ks_null[j] = kstest(u_perm, 'uniform').statistic

        p_emp = (np.sum(ks_null >= ks_obs) + 1) / (R + 1)
        results.append(dict(motif=motif, n_hits=n_hits,
                            ks_obs=ks_obs, p_emp=p_emp))

    # multiple-test correction
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df['q_value'] = multipletests(res_df.p_emp, method='fdr_bh')[1]
        res_df['significant'] = res_df.q_value < fdr
        res_df.sort_values('q_value', inplace=True)

    return res_df


def motif_co_occurence_test(df_hits: pd.DataFrame) -> pd.DataFrame:
    
    results = []
    #create a binary motif x gene matrix
    binary_matrix = df_hits.groupby(['gene', 'motif']).size().unstack(fill_value=0).astype(bool)
    motifs = binary_matrix.columns
    for m1, m2 in combinations(motifs, 2):
        a = binary_matrix[m1]
        b = binary_matrix[m2]

        #boolean mask to subset to genes only in at least one motif
        keep_mask = a | b
        a, b = a[keep_mask], b[keep_mask]

        #ct is the contingency table for fisher's exact test
        ct = np.zeros((2,2))    
        ct[1][1] = np.sum(a & b)
        ct[1][0] = np.sum(a & ~b)
        ct[0][1] = np.sum(~a & b)
        ct[0][0] = np.sum(~a & ~b)

        odds, p = fisher_exact(ct, alternative = "greater")
        results.append((m1, m2, ct[1][1], odds, p))

    df_pairs = pd.DataFrame(results, columns=['motif_A','motif_B','overlap','odds_ratio','p_raw'])
    df_pairs['q_value'] = multipletests(df_pairs.p_raw, method='fdr_bh')[1]
    df_pairs['significant'] = df_pairs.q_value < 0.05

    # 2) Fill the missing triangle and diagonals:
    #    -log₁₀(1) = 0 means “no evidence” by default.
    heat_df = df_pairs.pivot(index='motif_A', 
                        columns='motif_B', 
                        values='q_value')
    heat_matrix = -np.log10(heat_df.fillna(1.0))

    # 3) If you want a symmetric matrix (since (A,B) and (B,A) are the same),
    #    copy the upper‐triangle into the lower‐triangle:
    heat_matrix_full = heat_matrix.combine_first(heat_matrix.T)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heat_matrix_full,
        ax=ax,
        cmap='vlag_r',
        linewidths=0.5,
        linecolor='lightgray',
        cbar_kws={'label': '-log10 FDR-q'},
        square=True
    )
    ax.set_title("Motif-Motif Co-occurrence (Fisher's exact)")
    ax.set_xlabel('Motif B')
    ax.set_ylabel('Motif A')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()

    return df_pairs, fig

def mutual_information(df_hits: pd.DataFrame, min_delta = 50) -> pd.DataFrame:

    def calculate_min_distance(motif_pair, gene):
        A, B = motif_pair
        #filter df_hits to only the hits for that motif and that gene
        hits_A = df_hits.loc[(df_hits['gene'] == gene) & (df_hits['motif'] == A), 'distance'].values
        hits_B = df_hits.loc[(df_hits['gene'] == gene) & (df_hits['motif'] == B), 'distance'].values

        #changing shape for subtracing by broadcasting
        all_distances = np.abs(hits_A.reshape(-1,1) - hits_B.reshape(1,-1))
        min_distance = np.min(all_distances)
        return min_distance

    #binary presence matrix
    binary_matrix = df_hits.groupby(['gene', 'motif']).size().unstack(fill_value=0).astype(bool)
    motifs = list(binary_matrix.columns)
    genes = list(binary_matrix.index)

    motif_pairs = list(combinations(motifs, 2))
    num_pairs = len(motif_pairs)
    num_genes = len(genes)
    X = np.zeros((num_pairs, num_genes), dtype = int) #if both are present or not
    C = np.zeros((num_pairs, num_genes), dtype = int) #stores min abs distance between two motifs in a gene

    #fill X and C matrices
    for i, pair in enumerate(motif_pairs):
        A, B = pair
        for j, gene in enumerate(genes): 
            if binary_matrix.at[gene, A] and binary_matrix.at[gene, B]:
                X[i][j] = 1 
                delta = calculate_min_distance(pair, gene)
                C[i][j] = 1 if (delta <= min_delta) else 0
            else: #either or both of the motifs are absent
                continue
    
    #now for each motif pair we compute mutual information scores
    MI_values = np.zeros(num_pairs)
    for i in range (num_pairs):
        x = X[i, :]
        c = C[i, :]

        if np.all(x == 0):
            MI_values[i] = 0
        else:
            MI_values[i] = mutual_info_score(x, c)

    #build an empty MI matrix
    MI_matrix = pd.DataFrame(
        np.zeros((len(motifs), len(motifs))),
        index = motifs,
        columns = motifs
        )    
    for i, (A, B) in enumerate(motif_pairs):
        MI_matrix.at[A,B] = MI_values[i]
        MI_matrix.at[B,A] = MI_values[i]

    #heatmap to visualize the MI matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        MI_matrix,
        ax=ax,
        cmap='viridis',
        linewidths=0.5,
        linecolor='white',
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6, "color": "white"},
        cbar_kws={'label': 'MI(X;C)'},
        square=True,
        xticklabels=motifs,  
        yticklabels=motifs
    )

    ax.set_title(f'Motif × Motif Distance-Aware MI (Δ ≤ {min_delta} bp)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    return MI_matrix, fig
    
            
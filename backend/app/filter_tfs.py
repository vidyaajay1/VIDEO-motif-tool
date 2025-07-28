# filter_tfs.py

import csv
from app.pfm_loader import load_pfms

TF_MASTER_FILE = "data/tf_master_list.txt"  # Path to your TF list file (TSV)

def load_tf_master_list():
    tf_dict = {}
    with open(TF_MASTER_FILE, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            symbol = row.get("SYMBOL", "").strip()
            if symbol:
                tf_dict[symbol] = {
                    "flybase_id": row.get("FBID_KEY", "").strip(),
                    #"go_bio_process": row.get("GO_BIOLOGICAL_PROCESS", "").strip()
                }
    return tf_dict


def filter_tfs_from_gene_list(gene_list):
    tf_master = load_tf_master_list()
    pfms = load_pfms()
    results = []

    for gene in gene_list:
        if gene in tf_master:
            motif = pfms.get(gene)
            results.append({
                "symbol": gene,
                "flybase_id": tf_master[gene]["flybase_id"],
                "motif_id": motif["motif_id"] if motif else None,
                "pfm": motif["matrix"] if motif else None
            })

    return results

def filter_tfs_from_gene_list_old(gene_list):
    """
    gene_list: list of gene symbols (e.g., ["CG5828", "Sodh-1", "Men"])
    Returns: list of dicts with symbol, flybase_id, go_bio_process
    """
    tf_master = load_tf_master_list()
    results = []

    for gene in gene_list:
        if gene in tf_master:
            results.append({
                "symbol": gene,
                "flybase_id": tf_master[gene]["flybase_id"],
                #"go_bio_process": tf_master[gene]["go_bio_process"]
            })

    return results

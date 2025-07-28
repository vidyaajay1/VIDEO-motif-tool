import pandas as pd
import re
from typing import List, Dict, Tuple

def parse_gtf_attributes(attr_str: str) -> Dict[str, str]:
    """
    Parse the 9th‐column attributes of a GTF into a dict.
    e.g. 'gene_id "FBgn0037643"; gene_symbol "ScsbetaA"; ...'
    """
    # split on ; and strip
    attrs = {}
    for entry in attr_str.strip().split(';'):
        if not entry.strip():
            continue
        key, value = entry.strip().split(' ', 1)
        # remove surrounding quotes
        attrs[key] = value.strip().strip('"')
    return attrs

def build_fbgn_to_symbol_map(gtf_file: str) -> Dict[str, str]:
    """
    Scan a GTF and return a dict mapping gene_id (FBgn...) -> gene_symbol.
    """
    fbgn2sym = {}
    # only need columns: 0–7 are standard, 8 is attributes
    gtf = pd.read_csv(gtf_file, sep='\t', comment='#', header=None, usecols=[8], names=['attrs'])
    for attr_str in gtf['attrs']:
        attrs = parse_gtf_attributes(attr_str)
        gid = attrs.get('gene_id')
        sym = attrs.get('gene_symbol')
        if gid and sym:
            fbgn2sym[gid] = sym
    return fbgn2sym

def normalize_target_id(raw_id: str, fbgn2sym: Dict[str, str]) -> str:
    """
    Given a raw ID from the TSV, return a gene_symbol.
    If it's an FBgn found in fbgn2sym, map it; else assume it's already a symbol.
    """
    raw_id = raw_id.strip()
    if raw_id in fbgn2sym:
        return fbgn2sym[raw_id]
    else:
        # you could add more logic here (e.g. map protein IDs via another table)
        return raw_id

def subset_by_genes(
    df: pd.DataFrame,
    gtf_file: str,
    gene_list: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # build your FBgn→symbol map exactly as before
    fbgn2sym = build_fbgn_to_symbol_map(gtf_file)
    # normalize and subset
    df['gene_symbol'] = df['Target_ID'].apply(
        lambda x: normalize_target_id(x, fbgn2sym)
    )
    print(df.head())
    sub = df[df['gene_symbol'].isin(gene_list)].copy()
    pairs = (
        sub[['Query_ID','gene_symbol']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return sub, pairs


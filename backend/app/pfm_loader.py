# pfm_loader.py

import re

JASPAR_PFM_FILE = "data/jaspar_insect_pfms.txt"

def load_pfms():
    pfm_dict = {}
    with open(JASPAR_PFM_FILE, "r") as f:
        content = f.read()

    # Regex to extract each motif block
    motif_blocks = re.findall(
        r"MOTIF\s+(\S+)\s+(\S+)\s+.*?letter-probability matrix:.*?\n(.*?)(?=\n\n|\Z)", 
        content, 
        re.DOTALL
    )

    for motif_id, tf_name, matrix_block in motif_blocks:
        matrix_lines = [
            line.strip()
            for line in matrix_block.strip().split("\n")
            if line.strip() and not line.strip().startswith("URL")
        ]
        matrix = []
        for line in matrix_lines:
            row = [float(x) for x in re.split(r"\s+", line.strip())]
            matrix.append(row)

        pfm_dict[tf_name] = {
            "motif_id": motif_id,
            "matrix": matrix
        }


    return pfm_dict

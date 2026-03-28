// frontend/src/utils/parseMemeMotifs.ts

export interface ParsedMotif {
  name: string;
  matrix: number[][];
}

/**
 * Parses a MEME-format string (STREME .txt or JASPAR .meme)
 * and returns up to maxMotifs parsed motifs.
 * Each motif's name is taken from the last token of the MOTIF line
 * (e.g. "STREME-1" from "MOTIF 1-CCACCGCC STREME-1").
 */
export function parseMemeMotifs(
  text: string,
  maxMotifs = 10
): ParsedMotif[] {
  const lines = text.split(/\r?\n/);
  const results: ParsedMotif[] = [];

  let currentName: string | null = null;
  let collectingMatrix = false;
  let currentMatrix: number[][] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();

    // New MOTIF line — save previous if exists, start new
    if (line.startsWith("MOTIF ")) {
      // Save previous motif if we were collecting one
      if (currentName !== null && currentMatrix.length > 0) {
        results.push({ name: currentName, matrix: currentMatrix });
        if (results.length >= maxMotifs) break;
      }
      // Parse name: last token of the MOTIF line
      const tokens = line.split(/\s+/);
      currentName = tokens[tokens.length - 1];
      collectingMatrix = false;
      currentMatrix = [];
      continue;
    }

    // Start collecting rows after this header line
    if (line.startsWith("letter-probability matrix:")) {
      collectingMatrix = true;
      continue;
    }

    // Collect matrix rows
    if (collectingMatrix && /^\s*[\d\.\-]/.test(line)) {
      const row = line.trim().split(/\s+/).map(parseFloat);
      if (row.length === 4 && row.every((v) => !isNaN(v))) {
        currentMatrix.push(row);
      }
      continue;
    }

    // Any non-numeric line after matrix started = end of this matrix block
    if (collectingMatrix && line !== "" && !/^\s*[\d\.\-]/.test(line)) {
      collectingMatrix = false;
    }
  }

  // Don't forget the last motif in the file
  if (
    currentName !== null &&
    currentMatrix.length > 0 &&
    results.length < maxMotifs
  ) {
    results.push({ name: currentName, matrix: currentMatrix });
  }

  return results;
}
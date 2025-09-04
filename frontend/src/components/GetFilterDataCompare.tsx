import React, { useState, useEffect, useCallback } from "react";
const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

type Props = {
  sessionId: string;
  scanComplete: boolean;
  scanVersion: number;
  onDone: () => void; // call when filter+score is done (to goNext)
  onError: (msg: string) => void;
};

const GetFilterDataCompare: React.FC<Props> = ({
  sessionId,
  scanComplete,
  scanVersion,
  onDone,
  onError,
}) => {
  // mode: shared uploads for both lists vs per-dataset A/B uploads
  const [useShared, setUseShared] = useState(true);

  // shared files
  const [atacShared, setAtacShared] = useState<File | null>(null);
  const [chipShared, setChipShared] = useState<File | null>(null);

  // per-list files
  const [atacA, setAtacA] = useState<File | null>(null);
  const [chipA, setChipA] = useState<File | null>(null);
  const [atacB, setAtacB] = useState<File | null>(null);
  const [chipB, setChipB] = useState<File | null>(null);

  const [filterComplete, setFilterComplete] = useState(false);

  // reset when scan changes
  useEffect(() => {
    setFilterComplete(false);
    setAtacShared(null);
    setChipShared(null);
    setAtacA(null);
    setChipA(null);
    setAtacB(null);
    setChipB(null);
  }, [sessionId, scanComplete, scanVersion]);

  const canSubmit = useShared
    ? !!(atacShared || chipShared)
    : !!(atacA || chipA || atacB || chipB);

  const handleSubmit = useCallback(async () => {
    if (!scanComplete) return onError("Please complete motif scan first");
    if (!sessionId) return onError("Missing session_id");
    if (!canSubmit) return onError("Upload at least one ATAC/ChIP file");

    try {
      const fd = new FormData();
      fd.append("session_id", sessionId);

      if (useShared) {
        if (atacShared) fd.append("atac_bed_shared", atacShared);
        if (chipShared) fd.append("chip_bed_shared", chipShared);
      } else {
        if (atacA) fd.append("atac_bed_a", atacA);
        if (chipA) fd.append("chip_bed_a", chipA);
        if (atacB) fd.append("atac_bed_b", atacB);
        if (chipB) fd.append("chip_bed_b", chipB);
      }

      const res = await fetch(`${API_BASE}/filter-motif-hits-batch`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());

      setFilterComplete(true);
      onDone(); // advance to the next step
    } catch (e: any) {
      onError(e.message ?? String(e));
    }
  }, [
    scanComplete,
    sessionId,
    canSubmit,
    useShared,
    atacShared,
    chipShared,
    atacA,
    chipA,
    atacB,
    chipB,
    onError,
    onDone,
  ]);

  return (
    <div
      className={`text-center mb-3 mt-3 ${
        !scanComplete ? "bg-light text-muted" : ""
      }`}
    >
      <h4>Optional: Add chromatin accessibility/binding data for both lists</h4>

      <div className="form-check form-switch d-inline-flex gap-2 align-items-center my-3">
        <input
          id="use-shared"
          className="form-check-input"
          type="checkbox"
          checked={useShared}
          onChange={(e) => setUseShared(e.target.checked)}
          disabled={!scanComplete}
        />
        <label className="form-check-label" htmlFor="use-shared">
          Use the same ATAC/ChIP files for both lists
        </label>
      </div>

      {useShared ? (
        <div className="container" style={{ maxWidth: 850 }}>
          <div className="row mb-3 align-items-center">
            <div className="col">
              <label className="form-label">Shared ATAC-seq peaks (.bed)</label>
            </div>
            <div className="col-auto">
              <input
                type="file"
                accept=".bed"
                className="form-control form-control-sm"
                onChange={(e) => setAtacShared(e.target.files?.[0] ?? null)}
                disabled={!scanComplete}
              />
            </div>
          </div>
          <div className="row align-items-center">
            <div className="col">
              <label className="form-label">Shared ChIP-seq peaks (.bed)</label>
            </div>
            <div className="col-auto">
              <input
                type="file"
                accept=".bed"
                className="form-control form-control-sm"
                onChange={(e) => setChipShared(e.target.files?.[0] ?? null)}
                disabled={!scanComplete}
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="container" style={{ maxWidth: 900 }}>
          <div className="row fw-bold mt-2">
            <div className="col">List A</div>
            <div className="col">List B</div>
          </div>
          <div className="row align-items-center mt-1">
            <div className="col">
              <input
                type="file"
                accept=".bed"
                className="form-control form-control-sm"
                placeholder="ATAC A"
                onChange={(e) => setAtacA(e.target.files?.[0] ?? null)}
                disabled={!scanComplete}
              />
              <small className="text-muted">ATAC A</small>
            </div>
            <div className="col">
              <input
                type="file"
                accept=".bed"
                className="form-control form-control-sm"
                placeholder="ATAC B"
                onChange={(e) => setAtacB(e.target.files?.[0] ?? null)}
                disabled={!scanComplete}
              />
              <small className="text-muted">ATAC B</small>
            </div>
          </div>
          <div className="row align-items-center mt-2">
            <div className="col">
              <input
                type="file"
                accept=".bed"
                className="form-control form-control-sm"
                placeholder="ChIP A"
                onChange={(e) => setChipA(e.target.files?.[0] ?? null)}
                disabled={!scanComplete}
              />
              <small className="text-muted">ChIP A</small>
            </div>
            <div className="col">
              <input
                type="file"
                accept=".bed"
                className="form-control form-control-sm"
                placeholder="ChIP B"
                onChange={(e) => setChipB(e.target.files?.[0] ?? null)}
                disabled={!scanComplete}
              />
              <small className="text-muted">ChIP B</small>
            </div>
          </div>
        </div>
      )}

      <button
        type="button"
        className={`btn ${filterComplete ? "btn-success" : "btn-primary"} mt-3`}
        onClick={handleSubmit}
        disabled={!scanComplete || !canSubmit}
      >
        {filterComplete ? "✅ Filtered!" : "🧪 Filter & Score (Batch)"}
      </button>
    </div>
  );
};

export default GetFilterDataCompare;

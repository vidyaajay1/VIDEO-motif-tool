import React, { useState, useCallback } from "react";
import { UserMotif } from "./GetMotifInput";
import InfoTip from "./InfoTip";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
export interface ScanFIMOProps {
  dataId: string | null;
  inputWindow: number;
  validMotifs: UserMotif[];
  selectedStremeMotifs: Array<{ id: string; name?: string; color: string }>;
  discoveredMotifs: Array<{ id: string; name: string; pwm: number[][] }>;
  fetchJSON: (
    url: string,
    options: RequestInit,
    onError: (msg: string) => void
  ) => Promise<any>;
  onError: (msg: string) => void;
  scanComplete: boolean;
  onScanComplete?: () => void;
}

const ScanFIMO: React.FC<ScanFIMOProps> = ({
  dataId,
  inputWindow,
  validMotifs,
  selectedStremeMotifs,
  discoveredMotifs,
  fetchJSON,
  onError,
  scanComplete,
  onScanComplete,
}) => {
  const [fimoThreshold, setFimoThreshold] = useState<string>("0.005");

  const handleMotifScan = useCallback(async () => {
    if (!dataId) {
      onError("Please process genomic input first");
      return;
    }

    const allMotifs: UserMotif[] = [
      ...validMotifs,
      ...selectedStremeMotifs.map((sel) => {
        const d = discoveredMotifs.find((m) => m.id === sel.id);
        return {
          type: "pwm",
          iupac: "",
          pwm: d?.pwm ?? [[0.25, 0.25, 0.25, 0.25]],
          name: sel.name || d?.name || sel.id,
          color: sel.color,
        } as UserMotif;
      }),
    ];

    if (!allMotifs.length) {
      onError("Please provide or select at least one motif");
      return;
    }

    const fd = new FormData();
    allMotifs.forEach((m) => {
      const payload = {
        type: m.type,
        data: m.type === "iupac" ? m.iupac : m.type === "pwm" ? m.pwm : m.pcm,
        name: m.name,
        color: m.color,
      };
      fd.append("motifs", JSON.stringify(payload));
    });
    fd.append("window", String(inputWindow));
    fd.append("fimo_threshold", fimoThreshold);
    fd.append("data_id", dataId);

    const json = await fetchJSON(
      `${API_BASE}/get-motif-hits`,
      { method: "POST", body: fd },
      onError
    );
    if (!json) return;

    onError("Motif scan completed and saved to backend.");
    onScanComplete?.();
  }, [
    dataId,
    validMotifs,
    selectedStremeMotifs,
    discoveredMotifs,
    inputWindow,
    fimoThreshold,
    fetchJSON,
    onError,
    onScanComplete,
  ]);

  return (
    <div className="mt-3 text-center">
      <h4 className="text-center mb-3 mt-3">
        Scan for motif hits
        <InfoTip
          text="This is where we scan the genomic regions for motif hits. Choose a p-value threshold for the FIMO search algorithm."
          placement="right"
          id="genomic-input-info"
        />
      </h4>
      <div className="form-group mb-2 d-inline-block text-left">
        <label htmlFor="fimo-threshold" className="form-label">
          FIMO Threshold:
        </label>
        <input
          id="fimo-threshold"
          type="number"
          step="0.0001"
          min="0"
          max="1"
          value={fimoThreshold}
          onChange={(e) => setFimoThreshold(e.target.value)}
          className="form-control d-inline-block ml-2"
          style={{ width: "100px" }}
          disabled={!dataId} // only disable if no dataId
        />
      </div>
      <div className="mt-2">
        <button
          type="button"
          className={`btn ${scanComplete ? "btn-success" : "btn-primary"}`}
          onClick={handleMotifScan}
          disabled={!dataId}
        >
          🔍 {scanComplete ? "Re-scan Motifs" : "Scan Motifs (FIMO)"}
        </button>
      </div>
      <div>
        <small className="form-text text-muted mt-10 ms-10">
          FIMO: Grant et al. <i>Bioinformatics</i> 2011
        </small>
      </div>
      {!dataId && (
        <small className="form-text text-muted">
          Please process genomic input to enable scanning.
        </small>
      )}
    </div>
  );
};

export default ScanFIMO;

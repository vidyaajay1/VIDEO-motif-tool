import React, { useState, useCallback, useEffect } from "react";
const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
import InfoTip from "./InfoTip";
import { useMotifViewer } from "../context/MotifViewerContext";

export interface GetFilterDataProps {
  dataId: string | null;
  scanComplete: boolean;
  scanVersion: number;
  onFilteredOverview: (overviewUrl: string) => void;
  onError: (msg: string) => void;
}

const GetFilterData: React.FC<GetFilterDataProps> = ({
  dataId,
  scanComplete,
  scanVersion,
  onFilteredOverview,
  onError,
}) => {
  const [atacBedFile, setAtacBedFile] = useState<File | null>(null);
  const [chipBedFile, setChipBedFile] = useState<File | null>(null);
  const [showDownload, setShowDownload] = useState<boolean>(false);
  const [filterComplete, setFilterComplete] = useState(false);
  const { singleTopHitsReady, setSingleTopHitsReady } = useMotifViewer();

  // ← Reset filtered state whenever the underlying data changes
  // whenever we get a brand new scanVersion (or dataId/scanComplete changes),
  // clear out the “Filtered!” state so the button goes back to primary
  useEffect(() => {
    setFilterComplete(false);
    setShowDownload(false);
  }, [dataId, scanComplete, scanVersion]);
  const handleFilterMotifHits = useCallback(async () => {
    if (!scanComplete || !dataId) {
      onError("Please complete motif scan first");
      return;
    }
    // only require at least one
    if (!atacBedFile && !chipBedFile) {
      onError("Please upload at least one of ATAC or ChIP BED files");
      return;
    }

    try {
      // 1) Filter & Score only
      const form1 = new FormData();
      form1.append("data_id", dataId);
      if (atacBedFile) {
        form1.append("atac_bed", atacBedFile);
      }
      if (chipBedFile) {
        form1.append("chip_bed", chipBedFile);
      }

      const res1 = await fetch(`${API_BASE}/filter-motif-hits`, {
        method: "POST",
        body: form1,
      });
      if (!res1.ok) {
        const err = await res1.text();
        throw new Error(err);
      }

      // indicate success & show TSV download link
      setSingleTopHitsReady(true);
      setFilterComplete(true);
    } catch (e: any) {
      onError(e.message);
    }
  }, [dataId, scanComplete, atacBedFile, chipBedFile, onError]);

  // Always render, but disable controls if scan not complete
  return (
    <div
      className={`text-center mb-3 mt-3 ${
        !scanComplete ? "bg-light text-muted" : ""
      }`}
    >
      <h4 className="">
        Optional: Add chromatin accessibility/binding data
        <InfoTip
          text="We'll use this data to find motif hits in peak regions. This is completely optional, skip if you don't have any data."
          placement="right"
          id="genomic-input-info"
        />
      </h4>
      <div className="container mt-3" style={{ maxWidth: 850 }}>
        <div className="row mb-3 align-items-center">
          <div className="col">
            <label htmlFor="atac-bed" className="form-label">
              BED file with open chromatin regions (ATAC-seq peaks):
            </label>
          </div>
          <div className="col-auto">
            <input
              id="atac-bed"
              type="file"
              accept=".bed"
              className="form-control form-control-sm "
              onChange={(e) => setAtacBedFile(e.target.files?.[0] ?? null)}
              disabled={!scanComplete}
            />
          </div>
        </div>

        <div className="row align-items-center">
          <div className="col">
            <label htmlFor="chip-bed" className="form-label">
              BED file with binding peak regions (ChIP-seq peaks):
            </label>
          </div>
          <div className="col-auto">
            <input
              id="chip-bed"
              type="file"
              accept=".bed"
              className="form-control form-control-sm"
              onChange={(e) => setChipBedFile(e.target.files?.[0] ?? null)}
              disabled={!scanComplete}
            />
          </div>
        </div>
      </div>
      <button
        type="button"
        className={`btn ${filterComplete ? "btn-success" : "btn-primary"} mt-3`}
        onClick={handleFilterMotifHits}
        disabled={!scanComplete || (!atacBedFile && !chipBedFile)}
      >
        {filterComplete ? "✅ Filtered!" : "🧪 Filter & Score Motif Hits"}
      </button>

      {filterComplete && (
        <div className="mt-2 text-success">
          Motif hits have been filtered and scored!
        </div>
      )}

      {filterComplete && dataId && (
        <a
          href={`${API_BASE}/download-top-hits/${dataId}`}
          download
          className="btn btn-outline-success mt-3 ml-2"
        >
          ⬇ Download Top Hits
        </a>
      )}

      {!scanComplete && (
        <div className="mt-2">
          <small>Please complete the motif scan to enable filters.</small>
        </div>
      )}
    </div>
  );
};

export default GetFilterData;

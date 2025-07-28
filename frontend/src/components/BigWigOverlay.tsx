// BigWigOverlay.tsx
import React, { useState, useCallback } from "react";

export interface BigWigOverlayProps {
  dataId: string | null;
  inputWindow: number;
  peakList: string[];
  apiBase: string;
}

const BigWigOverlay: React.FC<BigWigOverlayProps> = ({
  dataId,
  inputWindow,
  peakList,
  apiBase,
}) => {
  const [gene, setGene] = useState("");
  const [bigwigs, setBigwigs] = useState<File[]>([]);
  const [trackInfo, setTrackInfo] = useState<string[]>([]);
  const [overlayImageUrl, setOverlayImageUrl] = useState<string | null>(null);

  const disabled = !dataId;

  const handleOverlaySubmit = useCallback(async () => {
    if (disabled) return;
    const infoOK = trackInfo.every((s) => {
      const [n, c] = s.split("|");
      return n && c;
    });
    if (!gene || bigwigs.length === 0 || !infoOK) {
      alert("Fill gene, tracks, names & colours");
      return;
    }
    const fd = new FormData();
    fd.append("data_id", dataId);
    fd.append("gene", gene);
    fd.append("window", String(inputWindow));
    bigwigs.forEach((f) => fd.append("bigwigs", f));
    trackInfo.forEach((s) => fd.append("chip_tracks", s));

    try {
      const res = await fetch(`${apiBase}/plot-chip-overlay`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      setOverlayImageUrl(URL.createObjectURL(await res.blob()));
    } catch (e: any) {
      alert(e.message);
    }
  }, [dataId, gene, bigwigs, trackInfo, inputWindow, apiBase, disabled]);

  return (
    <div className={`card mt-4 ${disabled ? "bg-light text-muted" : ""}`}>
      <div className="card-body">
        <h5 className="card-title">Overlay ChIP/ATAC BigWig Tracks</h5>

        <div className="mb-3">
          <label className="form-label">Choose Peak/Gene:</label>
          <select
            className="form-select"
            value={gene}
            onChange={(e) => setGene(e.target.value)}
            disabled={disabled}
          >
            <option value="">-- pick one --</option>
            {peakList.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </div>

        <div className="mb-3">
          <input
            type="file"
            className="form-control"
            multiple
            accept=".bw,.bigwig"
            onChange={(e) => {
              if (disabled) return;
              const files = Array.from(e.target.files ?? []);
              if (!files.length) return;
              setBigwigs((prev) => [...prev, ...files]);
              setTrackInfo((prev) => [...prev, ...files.map(() => "|")]);
              e.target.value = "";
            }}
            disabled={disabled}
          />
        </div>

        {bigwigs.map((file, idx) => (
          <div key={idx} className="d-flex align-items-center mb-2">
            <small
              className="flex-grow-1 text-truncate"
              style={{ maxWidth: 160 }}
            >
              {file.name}
            </small>
            <input
              type="text"
              className="form-control form-control-sm me-2"
              placeholder="Track name"
              value={trackInfo[idx]?.split("|")[0] ?? ""}
              onChange={(e) => {
                const [, c = ""] = (trackInfo[idx] || "|").split("|");
                setTrackInfo((t) => {
                  const copy = [...t];
                  copy[idx] = `${e.target.value}|${c}`;
                  return copy;
                });
              }}
              disabled={disabled}
            />
            <input
              type="color"
              className="form-control form-control-color form-control-sm me-2"
              value={trackInfo[idx]?.split("|")[1] || "#000000"}
              onChange={(e) => {
                const [n = ""] = (trackInfo[idx] || "|").split("|");
                setTrackInfo((t) => {
                  const copy = [...t];
                  copy[idx] = `${n}|${e.target.value}`;
                  return copy;
                });
              }}
              disabled={disabled}
            />
            <button
              className="btn btn-sm btn-outline-danger"
              onClick={() => {
                if (disabled) return;
                setBigwigs((b) => b.filter((_, i) => i !== idx));
                setTrackInfo((t) => t.filter((_, i) => i !== idx));
              }}
              disabled={disabled}
            >
              ×
            </button>
          </div>
        ))}

        <button
          className="btn btn-primary"
          onClick={handleOverlaySubmit}
          disabled={disabled}
        >
          Generate Overlay Plot
        </button>

        {overlayImageUrl && (
          <div className="card mt-4">
            <div className="card-body text-center">
              <img
                src={overlayImageUrl}
                alt="ChIP/ATAC overlay"
                className="img-fluid"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BigWigOverlay;

import React, { useMemo, useState, useCallback } from "react";
import Plot from "react-plotly.js";

type PlotlyJSON = { data: any[]; layout: any; frames?: any[] };

export interface BigWigOverlayCompareProps {
  sessionId: string | null;
  inputWindow: number;
  // map: label → peak list for that label
  peakListsByLabel: Record<string, string[]>;
  labels: string[]; // e.g., ["List A", "List B"]
  apiBase: string;
}

const BigWigOverlayCompare: React.FC<BigWigOverlayCompareProps> = ({
  sessionId,
  inputWindow,
  peakListsByLabel,
  labels,
  apiBase,
}) => {
  const [label, setLabel] = useState<string>(labels[0] || "");
  const [gene, setGene] = useState("");
  const [bigwigs, setBigwigs] = useState<File[]>([]);
  const [trackInfo, setTrackInfo] = useState<string[]>([]);
  const [fig, setFig] = useState<PlotlyJSON | null>(null);
  const [loading, setLoading] = useState(false);

  const disabled = !sessionId;

  const peaks = useMemo(
    () => (label ? peakListsByLabel[label] || [] : []),
    [label, peakListsByLabel]
  );

  const handleOverlaySubmit = useCallback(async () => {
    if (disabled) return;

    const infoOK = trackInfo.every((s) => {
      const [n, c] = s.split("|");
      return n && c;
    });

    if (!label || !gene || bigwigs.length === 0 || !infoOK) {
      alert(
        "Pick a list, choose a peak, add files, and set track names & colors."
      );
      return;
    }

    const fd = new FormData();
    fd.append("session_id", String(sessionId));
    fd.append("label", label);
    fd.append("gene", gene);
    fd.append("window", String(inputWindow));
    bigwigs.forEach((f) => fd.append("bigwigs", f));
    trackInfo.forEach((s) => fd.append("chip_tracks", s));

    setLoading(true);
    setFig(null);

    try {
      const res = await fetch(`${apiBase}/plot-chip-overlay-compare`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        let msg = await res.text();
        try {
          const j = JSON.parse(msg);
          if (j?.detail) msg = j.detail;
        } catch {}
        throw new Error(msg || "Request failed");
      }

      const payload = await res.json();
      const figJSON = JSON.parse(payload.chip_overlay_plot) as PlotlyJSON;
      figJSON.layout = { ...figJSON.layout, autosize: true };
      setFig(figJSON);
    } catch (e: any) {
      alert(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }, [
    sessionId,
    label,
    gene,
    bigwigs,
    trackInfo,
    inputWindow,
    apiBase,
    disabled,
  ]);

  return (
    <div className={`card mt-4 ${disabled ? "bg-light text-muted" : ""}`}>
      <div className="card-body">
        <h5 className="card-title">
          Overlay ChIP/ATAC BigWig Tracks (Compare)
        </h5>

        <div className="row g-3">
          <div className="col-md-4">
            <label className="form-label">List</label>
            <select
              className="form-select"
              value={label}
              onChange={(e) => {
                setLabel(e.target.value);
                setGene("");
              }}
              disabled={disabled}
            >
              {labels.map((lb) => (
                <option key={lb} value={lb}>
                  {lb}
                </option>
              ))}
            </select>
          </div>

          <div className="col-md-8">
            <label className="form-label">Peak/Gene</label>
            <select
              className="form-select"
              value={gene}
              onChange={(e) => setGene(e.target.value)}
              disabled={disabled || !label}
            >
              <option value="">-- pick one --</option>
              {peaks.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-3">
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
              e.currentTarget.value = "";
            }}
            disabled={disabled}
          />
        </div>

        {bigwigs.map((file, idx) => (
          <div key={idx} className="d-flex align-items-center mt-2">
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
                const name = e.target.value;
                setTrackInfo((t) => {
                  const copy = [...t];
                  copy[idx] = `${name}|${c}`;
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
                const color = e.target.value;
                setTrackInfo((t) => {
                  const copy = [...t];
                  copy[idx] = `${n}|${color}`;
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
          className="btn btn-primary mt-3"
          onClick={handleOverlaySubmit}
          disabled={disabled || loading}
        >
          {loading ? "Generating…" : "Generate Overlay Plot"}
        </button>

        {fig && (
          <div className="card mt-4">
            <div className="card-body">
              <div
                style={{
                  position: "relative",
                  width: "100%",
                  height: `${fig.layout?.height ?? 520}px`,
                }}
              >
                <Plot
                  data={fig.data}
                  layout={{ ...fig.layout, autosize: true }}
                  frames={fig.frames}
                  useResizeHandler
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                  }}
                  config={{
                    responsive: true,
                    displaylogo: false,
                    toImageButtonOptions: {
                      filename: `chip_overlay_${label}_${gene}`,
                    },
                    modeBarButtonsToRemove: ["lasso2d", "select2d"],
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BigWigOverlayCompare;

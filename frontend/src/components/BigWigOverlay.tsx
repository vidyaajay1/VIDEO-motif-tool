import React, { useState, useCallback } from "react";
import Plot from "react-plotly.js";

export interface BigWigOverlayProps {
  dataId: string | null;
  inputWindow: number;
  peakList: string[];
  apiBase: string;
}

type PlotlyJSON = {
  data: any[];
  layout: any;
  frames?: any[];
};

/* ───────── helpers for async RQ flow ───────── */
async function postJSON(url: string, body: FormData) {
  const res = await fetch(url, { method: "POST", body });
  if (!res.ok) {
    let msg = await res.text();
    try {
      const j = JSON.parse(msg);
      if (j?.detail) msg = j.detail;
    } catch {}
    throw new Error(msg || "Request failed");
  }
  return res.json();
}

async function getJSON(url: string) {
  const res = await fetch(url);
  if (!res.ok) {
    let msg = await res.text();
    try {
      const j = JSON.parse(msg);
      if (j?.detail) msg = j.detail;
    } catch {}
    throw new Error(msg || "Request failed");
  }
  return res.json();
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitForJob(
  apiBase: string,
  jobId: string,
  timeoutMs = 10 * 60_000,
  intervalMs = 1500
) {
  const start = Date.now();
  while (true) {
    const js = await getJSON(`${apiBase}/jobs/${encodeURIComponent(jobId)}`);
    if (js.status === "finished") return js;
    if (js.status === "failed") throw new Error(js.error || "Job failed");
    if (Date.now() - start > timeoutMs)
      throw new Error("Timed out waiting for job");
    await sleep(intervalMs);
  }
}
/* ───────────────────────────────────────────── */

const BigWigOverlay: React.FC<BigWigOverlayProps> = ({
  dataId,
  inputWindow,
  peakList,
  apiBase,
}) => {
  const [gene, setGene] = useState("");
  const [bigwigs, setBigwigs] = useState<File[]>([]);
  const [trackInfo, setTrackInfo] = useState<string[]>([]);
  const [fig, setFig] = useState<PlotlyJSON | null>(null);
  const [loading, setLoading] = useState(false);

  const disabled = !dataId;

  const handleOverlaySubmit = useCallback(async () => {
    if (disabled) return;

    const infoOK = trackInfo.every((s) => {
      const [n, c] = s.split("|");
      return n && c;
    });

    if (!gene || bigwigs.length === 0 || !infoOK) {
      alert("Fill gene, add files, and set track names & colors.");
      return;
    }

    const fd = new FormData();
    fd.append("data_id", String(dataId));
    fd.append("gene", gene);
    fd.append("window", String(inputWindow));
    // if you add these controls later, set them the same way:
    // fd.append("per_motif_pvals_json", "");
    // fd.append("min_score_bits", "0");
    bigwigs.forEach((f) => fd.append("bigwigs", f)); // multiple files
    trackInfo.forEach((s) => fd.append("chip_tracks", s)); // "Name|#RRGGBB"

    setLoading(true);
    setFig(null);

    try {
      // 1) enqueue
      const { job_id } = await postJSON(`${apiBase}/plot-chip-overlay`, fd);

      // 2) wait
      await waitForJob(apiBase, job_id);

      // 3) fetch artifact
      const payload = await getJSON(
        `${apiBase}/plots/chip-overlay/${encodeURIComponent(
          String(dataId)
        )}/${encodeURIComponent(gene)}`
      );

      // backend returns an object for `chip_overlay_plot`
      const plotObj = payload.chip_overlay_plot; // already parsed JSON (object)
      const figJSON = (
        typeof plotObj === "string" ? JSON.parse(plotObj) : plotObj
      ) as PlotlyJSON;

      // Make layout responsive-friendly
      figJSON.layout = { ...figJSON.layout, autosize: true };

      setFig(figJSON);
    } catch (e: any) {
      alert(e.message ?? String(e));
    } finally {
      setLoading(false);
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
              e.currentTarget.value = "";
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
          className="btn btn-primary"
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
                    toImageButtonOptions: { filename: `chip_overlay_${gene}` },
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

export default BigWigOverlay;

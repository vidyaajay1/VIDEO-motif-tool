import React, { useState, useCallback, useEffect } from "react";
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

type TrackMeta = {
  path: string;
  label: string;
  color: string;
  sha1: string;
};
type TrackRegistry = Record<string, TrackMeta>; // track_id -> meta

/* ───────── helpers ───────── */
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
/* ─────────────────────────── */

const BigWigOverlay: React.FC<BigWigOverlayProps> = ({
  dataId,
  inputWindow,
  peakList,
  apiBase,
}) => {
  const [gene, setGene] = useState("");
  // New: pending uploads (only used to register once)
  const [bigwigs, setBigwigs] = useState<File[]>([]);
  const [trackInfo, setTrackInfo] = useState<string[]>([]); // "Name|#RRGGBB" for pending uploads

  // New: persisted registry and selection
  const [registry, setRegistry] = useState<TrackRegistry>({});
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const [fig, setFig] = useState<PlotlyJSON | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const disabled = !dataId;

  // Load persisted tracks when dataId changes
  useEffect(() => {
    (async () => {
      if (!dataId) {
        setRegistry({});
        setSelectedIds([]);
        return;
      }
      try {
        const js = await getJSON(
          `${apiBase}/tracks/${encodeURIComponent(String(dataId))}`
        );
        const tracks: TrackRegistry = js?.tracks || {};
        setRegistry(tracks);
        setSelectedIds(Object.keys(tracks)); // default: select all
      } catch {
        setRegistry({});
        setSelectedIds([]);
      }
    })();
  }, [dataId, apiBase]);

  // Register new tracks (once), then refresh registry
  const handleRegister = useCallback(async () => {
    if (disabled) return;
    if (bigwigs.length === 0 || trackInfo.length !== bigwigs.length) {
      alert("Add .bw files and set track name & color for each.");
      return;
    }
    // validate label|#RRGGBB quickly
    if (
      !trackInfo.every((s) => {
        const [n, c] = s.split("|");
        return Boolean(n) && Boolean(c);
      })
    ) {
      alert("Each track needs a name and a color (Label|#RRGGBB).");
      return;
    }

    const fd = new FormData();
    fd.append("data_id", String(dataId));
    bigwigs.forEach((f) => fd.append("bigwigs", f));
    trackInfo.forEach((s) => fd.append("tracknames", s)); // backend expects "tracknames"

    setSaving(true);
    try {
      await postJSON(`${apiBase}/tracks/register`, fd);
      // clear the pending uploads UI
      setBigwigs([]);
      setTrackInfo([]);

      // reload registry
      const js = await getJSON(
        `${apiBase}/tracks/${encodeURIComponent(String(dataId))}`
      );
      const tracks: TrackRegistry = js?.tracks || {};
      setRegistry(tracks);
      setSelectedIds(Object.keys(tracks)); // select all by default
    } catch (e: any) {
      alert(e.message ?? String(e));
    } finally {
      setSaving(false);
    }
  }, [disabled, bigwigs, trackInfo, dataId, apiBase]);

  // Enqueue plot using persisted track_ids_json
  const handleOverlaySubmit = useCallback(async () => {
    if (disabled) return;
    if (!gene) {
      alert("Pick a gene/peak.");
      return;
    }
    if (selectedIds.length === 0) {
      alert("Select at least one registered track.");
      return;
    }

    const fd = new FormData();
    fd.append("data_id", String(dataId));
    fd.append("gene", gene);
    fd.append("window", String(inputWindow));
    fd.append("track_ids_json", JSON.stringify(selectedIds));

    setLoading(true);
    setFig(null);
    try {
      const { job_id } = await postJSON(`${apiBase}/plot-chip-overlay`, fd);
      await waitForJob(apiBase, job_id);

      const payload = await getJSON(
        `${apiBase}/plots/chip-overlay/${encodeURIComponent(
          String(dataId)
        )}/${encodeURIComponent(gene)}`
      );
      const plotObj = payload.chip_overlay_plot;
      const figJSON = (
        typeof plotObj === "string" ? JSON.parse(plotObj) : plotObj
      ) as PlotlyJSON;
      figJSON.layout = { ...figJSON.layout, autosize: true };
      setFig(figJSON);
    } catch (e: any) {
      alert(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }, [disabled, gene, selectedIds, dataId, inputWindow, apiBase]);

  return (
    <div className={`card mt-4 ${disabled ? "bg-light text-muted" : ""}`}>
      <div className="card-body">
        <h5 className="card-title mb-4">Overlay ChIP/ATAC BigWig Tracks</h5>

        {/* 1. Add up to three tracks and process them */}
        <div className="mb-4">
          <div className="d-flex align-items-baseline justify-content-between mb-2">
            <span className="fw-semibold">
              1. Add up to three tracks and process them
            </span>
          </div>

          <div className="mb-2">
            <input
              type="file"
              className="form-control"
              multiple
              accept=".bw,.bigwig"
              onChange={(e) => {
                if (disabled) return;
                const files = Array.from(e.target.files ?? []);
                if (!files.length) return;
                setBigwigs((prev) => [...prev, ...files].slice(0, 3));
                setTrackInfo((prev) =>
                  [...prev, ...files.map(() => "|")].slice(0, 3)
                );
                e.currentTarget.value = "";
              }}
              disabled={disabled}
            />
          </div>

          {bigwigs.length > 0 && (
            <div className="mb-2">
              {bigwigs.map((file, idx) => (
                <div key={idx} className="d-flex align-items-center mb-2">
                  <small
                    className="flex-grow-1 text-truncate"
                    style={{ maxWidth: 200 }}
                  >
                    {file.name}
                  </small>

                  {/* Optional inline name/color (kept from your original) */}
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
                    aria-label="Remove file"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            className="btn btn-outline-secondary"
            onClick={handleRegister}
            disabled={disabled || saving || bigwigs.length === 0}
            title="Upload & persist these tracks for this dataId"
          >
            {saving ? "Processing…" : "Process Tracks"}
          </button>
        </div>

        {/* 2. Choose the tracks you want to visualize */}
        <div className="mb-4">
          <div className="fw-semibold mb-2">
            2. Choose the tracks you want to visualize
          </div>

          {Object.keys(registry).length === 0 ? (
            <div className="form-text">
              No tracks yet. Upload and click Process Tracks above.
            </div>
          ) : (
            <div className="d-flex flex-column gap-1">
              {Object.entries(registry).map(([tid, meta]) => (
                <label key={tid} className="d-flex align-items-center gap-2">
                  <input
                    type="checkbox"
                    className="form-check-input"
                    checked={selectedIds.includes(tid)}
                    onChange={(e) => {
                      setSelectedIds((prev) =>
                        e.target.checked
                          ? [...new Set([...prev, tid])]
                          : prev.filter((x) => x !== tid)
                      );
                    }}
                    disabled={disabled}
                  />
                  <span
                    className="badge text-bg-light"
                    style={{ border: "1px solid #ddd" }}
                  >
                    <span
                      style={{
                        display: "inline-block",
                        width: 12,
                        height: 12,
                        background: meta.color,
                        marginRight: 6,
                        verticalAlign: "middle",
                      }}
                    />
                    {meta.label}
                  </span>
                  <small
                    className="text-muted text-truncate"
                    style={{ maxWidth: 240 }}
                  >
                    {meta.path.split("/").pop()}
                  </small>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* 3. Pick gene */}
        <div className="mb-4">
          <div className="fw-semibold mb-2">3. Pick gene</div>
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

        {/* 4. Generate Plot */}
        <div className="mb-2">
          <div className="fw-semibold mb-2">4. Generate Plot</div>
          <button
            className="btn btn-primary"
            onClick={handleOverlaySubmit}
            disabled={disabled || loading || !gene || selectedIds.length === 0}
          >
            {loading ? "Generating…" : "Generate Overlay Plot"}
          </button>
        </div>

        {/* Figure */}
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

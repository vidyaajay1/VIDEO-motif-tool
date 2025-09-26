// BigWigOverlayCompare.tsx
import React, { useMemo, useState, useCallback, useEffect } from "react";
import Plot from "react-plotly.js";

type PlotlyJSON = { data: any[]; layout: any; frames?: any[] };

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

export interface BigWigOverlayCompareProps {
  sessionId: string | null;
  inputWindow: number;
  // map: label → peak list for that label
  peakListsByLabel: Record<string, string[]>;
  labels: string[]; // e.g., ["List A", "List B"]
  apiBase: string;
}

type TrackMeta = {
  path: string;
  label: string;
  color: string;
  sha1: string;
};
type TrackRegistry = Record<string, TrackMeta>; // track_id -> meta

const BigWigOverlayCompare: React.FC<BigWigOverlayCompareProps> = ({
  sessionId,
  inputWindow,
  peakListsByLabel,
  labels,
  apiBase,
}) => {
  const [label, setLabel] = useState<string>(labels[0] || "");
  const [gene, setGene] = useState("");

  // Pending uploads (for one-time registration)
  const [bigwigs, setBigwigs] = useState<File[]>([]);
  const [trackInfo, setTrackInfo] = useState<string[]>([]); // "Name|#RRGGBB"

  // Persisted registry + selection
  const [registry, setRegistry] = useState<TrackRegistry>({});
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const [fig, setFig] = useState<PlotlyJSON | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const disabled = !sessionId;

  const peaks = useMemo(
    () => (label ? peakListsByLabel[label] || [] : []),
    [label, peakListsByLabel]
  );

  // Load registry whenever namespace (sessionId+label) changes
  useEffect(() => {
    (async () => {
      if (!sessionId || !label) {
        setRegistry({});
        setSelectedIds([]);
        return;
      }
      try {
        const js = await getJSON(
          `${apiBase}/tracks-compare/${encodeURIComponent(
            String(sessionId)
          )}/${encodeURIComponent(label)}`
        );
        const tracks: TrackRegistry = js?.tracks || {};
        setRegistry(tracks);
        setSelectedIds(Object.keys(tracks)); // default: all
      } catch {
        setRegistry({});
        setSelectedIds([]);
      }
    })();
  }, [sessionId, label, apiBase]);

  // Register new tracks for this {sessionId,label}
  const handleRegister = useCallback(async () => {
    if (disabled) return;
    if (!label) {
      alert("Pick a list label first.");
      return;
    }
    if (bigwigs.length === 0 || trackInfo.length !== bigwigs.length) {
      alert("Add .bw files and set track name & color for each.");
      return;
    }
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
    fd.append("session_id", String(sessionId));
    fd.append("label", label);
    bigwigs.forEach((f) => fd.append("bigwigs", f));
    trackInfo.forEach((s) => fd.append("tracknames", s)); // backend expects "tracknames"

    setSaving(true);
    try {
      await postJSON(`${apiBase}/tracks/register-compare`, fd);

      // Clear pending uploads
      setBigwigs([]);
      setTrackInfo([]);

      // Reload registry
      const js = await getJSON(
        `${apiBase}/tracks-compare/${encodeURIComponent(
          String(sessionId)
        )}/${encodeURIComponent(label)}`
      );
      const tracks: TrackRegistry = js?.tracks || {};
      setRegistry(tracks);
      setSelectedIds(Object.keys(tracks));
    } catch (e: any) {
      alert(e.message ?? String(e));
    } finally {
      setSaving(false);
    }
  }, [disabled, bigwigs, trackInfo, sessionId, label, apiBase]);

  // Enqueue plot using persisted track_ids_json
  const handleOverlaySubmit = useCallback(async () => {
    if (disabled) return;
    if (!label || !gene) {
      alert("Pick a list and choose a peak.");
      return;
    }
    if (selectedIds.length === 0) {
      alert("Select at least one registered track.");
      return;
    }

    const fd = new FormData();
    fd.append("session_id", String(sessionId));
    fd.append("label", label);
    fd.append("gene", gene);
    fd.append("window", String(inputWindow));
    fd.append("track_ids_json", JSON.stringify(selectedIds));
    // If needed later:
    // fd.append("per_motif_pvals_json", "");
    // fd.append("min_score_bits", "0");

    setLoading(true);
    setFig(null);

    try {
      const { job_id } = await postJSON(
        `${apiBase}/plot-chip-overlay-compare`,
        fd
      );

      await waitForJob(apiBase, job_id);

      const json = await getJSON(
        `${apiBase}/plots/chip-overlay-compare/${encodeURIComponent(
          String(sessionId)
        )}/${encodeURIComponent(label)}/${encodeURIComponent(gene)}`
      );

      const plotObj = json.chip_overlay_plot;
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
  }, [disabled, sessionId, label, gene, selectedIds, inputWindow, apiBase]);

  return (
    <div className={`card mt-4 ${disabled ? "bg-light text-muted" : ""}`}>
      <div className="card-body">
        <h5 className="card-title">
          Overlay ChIP/ATAC BigWig Tracks (Compare)
        </h5>

        {/* List/Peak selectors */}
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

        {/* Registered tracks for this {sessionId, label} */}
        <div className="mt-3">
          <label className="form-label">
            Registered Tracks (persisted for this list)
          </label>
          {Object.keys(registry).length === 0 ? (
            <div className="form-text">
              No tracks yet for this list. Upload &amp; register below.
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

        {/* Upload + register (one-time per list) */}
        <div className="mt-3">
          <label className="form-label">Add new .bw files (optional)</label>
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

        <div className="d-flex gap-2 mt-3">
          <button
            className="btn btn-outline-secondary"
            onClick={handleRegister}
            disabled={disabled || saving || bigwigs.length === 0}
            title="Upload & persist these tracks for this list"
          >
            {saving ? "Registering…" : "Register Tracks"}
          </button>

          <button
            className="btn btn-primary"
            onClick={handleOverlaySubmit}
            disabled={
              disabled || loading || !label || !gene || selectedIds.length === 0
            }
          >
            {loading ? "Generating…" : "Generate Overlay Plot"}
          </button>
        </div>

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

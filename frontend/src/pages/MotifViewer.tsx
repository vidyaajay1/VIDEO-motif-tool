// MotifViewer.tsx
import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Container, Card, Nav, Spinner, Button } from "react-bootstrap";

import GenomicInput from "../components/GenomicInput";
import GetMotifInput from "../components/GetMotifInput";
import ScanFIMO from "../components/ScanFIMO";
import GetFilterData from "../components/GetFilterData";
import GetFilterDataCompare from "../components/GetFilterDataCompare";
import MotifOccurencePlot from "../components/MotifOccurencePlot";
import BigWigOverlay from "../components/BigWigOverlay";
import BigWigOverlayCompare from "../components/BigWigOverlayCompare";
import InfoTip from "../components/InfoTip";

import { useMotifViewer } from "../context/MotifViewerContext";

import CompareInputs from "../components/CompareInputs";
import MotifOccurrenceCompare from "../components/MotifOccurrenceCompare";
import type { FilterSettings } from "../types/FilterSettings";
import FiltersBar from "../components/FiltersBar";
const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const steps = [
  "Genomic Input",
  "Enter Motifs",
  "Scan with FIMO",
  "Set Filters",
  "Occurrence Overview",
  "Track Overlay",
];

type Step = 0 | 1 | 2 | 3 | 4 | 5;

function MotifViewer() {
  const [compareMode, setCompareMode] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [labelA, setLabelA] = useState("List A");
  const [labelB, setLabelB] = useState("List B");
  const [geneListFileA, setGeneListFileA] = useState<File | null>(null);
  const [geneListFileB, setGeneListFileB] = useState<File | null>(null);
  const [activeStep, setActiveStep] = useState<Step>(() => {
    const saved = sessionStorage.getItem("motifViewer.activeStep");
    return saved !== null ? (parseInt(saved, 10) as Step) : 0;
  });
  const goNext = () =>
    setActiveStep((s) => Math.min(steps.length - 1, s + 1) as Step);
  const goBack = () => setActiveStep((s) => Math.max(0, s - 1) as Step);

  const {
    dataType,
    setDataType,
    bedFile,
    setBedFile,
    geneListFile,
    setGeneListFile,
    inputWindow,
    setInputWindow,
    dataId,
    setDataId,
    peakList,
    setPeakList,
    motifs,
    setMotifs,
    scanComplete,
    setScanComplete,
    fimoThreshold,
    labelsByDataId,
    setLabelsByDataId,
    fetchJSON,
  } = useMotifViewer();

  // NEW: local state to hold Plotly JSON strings
  const [overviewFigureJson, setOverviewFigureJson] = useState<string | null>(
    null
  );
  const [filteredOverviewFigureJson, setFilteredOverviewFigureJson] = useState<
    string | null
  >(null);
  const [overviewFiguresByLabel, setOverviewFiguresByLabel] = useState<
    Record<string, string>
  >({});
  const [filteredFiguresByLabel, setFilteredFiguresByLabel] = useState<
    Record<string, string>
  >({});

  type OrderedPeaksByLabel = Record<string, string[]>;

  const [orderedPeaksByLabel, setOrderedPeaksByLabel] =
    useState<OrderedPeaksByLabel>({});

  const [scanVersion, setScanVersion] = useState<number>(() => {
    const saved = sessionStorage.getItem("motifViewer.scanVersion");
    return saved !== null ? parseInt(saved, 10) : 0;
  });
  useEffect(() => {
    sessionStorage.setItem("motifViewer.activeStep", String(activeStep));
  }, [activeStep]);
  useEffect(() => {
    sessionStorage.setItem("motifViewer.scanVersion", String(scanVersion));
  }, [scanVersion]);
  const updateMotif = (i: number, p: Partial<(typeof motifs)[0]>) =>
    setMotifs((ms) => ms.map((m, idx) => (idx === i ? { ...m, ...p } : m)));

  const addMotif = () =>
    setMotifs((ms) =>
      ms.length < 10
        ? [
            ...ms,
            {
              type: "iupac",
              iupac: "",
              pwm: [[0.25, 0.25, 0.25, 0.25]],
              pcm: [[1, 1, 1, 1]],
              name: "",
              color: "#d2a32bff",
            },
          ]
        : ms
    );

  const removeMotif = (i: number) =>
    setMotifs((ms) => ms.filter((_, idx) => idx !== i));

  const handleGetGenomicInputCompare = async () => {
    if (!geneListFileA || !geneListFileB)
      return alert("Upload both gene lists (A & B)");

    const fd = new FormData();
    fd.append("gene_list_file_a", geneListFileA);
    fd.append("gene_list_file_b", geneListFileB);
    fd.append("label_a", labelA);
    fd.append("label_b", labelB);
    fd.append("window_size", String(inputWindow));

    const json = await fetchJSON(
      `${API_BASE}/get-genomic-input-compare`,
      { method: "POST", body: fd },
      alert
    );
    if (json) {
      // { session_id, datasets: [{data_id,label,peak_list}, ...] }
      setSessionId(json.session_id);
      // for convenience you can retain the first data_id for components that still need one
      setDataId(json.datasets[0]?.data_id ?? null);
      setPeakList(json.datasets[0]?.peak_list ?? []);
      setScanComplete(false);
      const mapping: Record<string, string> = {};
      json.datasets.forEach((ds: any) => {
        mapping[ds.data_id] = ds.label;
      });
      setLabelsByDataId(mapping);
      // clear plots
      setOverviewFigureJson(null);
      setFilteredOverviewFigureJson(null);
      setOverviewFiguresByLabel({});
      setFilteredFiguresByLabel({});
    }
  };

  // process motifs
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validatedMotifs, setValidatedMotifs] = useState<typeof motifs>([]);
  const [processedSuccess, setProcessedSuccess] = useState(false);
  const [isProcessingMotifs, setIsProcessingMotifs] = useState(false);

  const handleProcessMotifs = async (): Promise<void> => {
    setProcessedSuccess(false);
    setValidationError(null);
    setIsProcessingMotifs(true);

    const fd = new FormData();
    motifs.forEach((m) => {
      const payload: Record<string, any> = {
        name: m.name,
        color: m.color,
        type: m.type,
      };
      if (m.type === "pwm") payload.pwm = m.pwm;
      else if (m.type === "pcm") payload.pcm = m.pcm;
      else payload.iupac = m.iupac;
      fd.append("motifs", JSON.stringify(payload));
    });

    try {
      if (compareMode) {
        if (!sessionId) return alert("Missing session_id");
        fd.append("session_id", sessionId);
        await fetchJSON(
          `${API_BASE}/validate-motifs-group`,
          { method: "POST", body: fd },
          setValidationError
        );
      } else {
        if (!dataId) return alert("Missing data_id");
        fd.append("data_id", dataId);
        await fetchJSON(
          `${API_BASE}/validate-motifs`,
          { method: "POST", body: fd },
          setValidationError
        );
      }
      setValidatedMotifs([...motifs]);
      setProcessedSuccess(true);
    } catch (err: any) {
      setValidationError("Network error: " + err.message);
    } finally {
      setIsProcessingMotifs(false);
    }
  };

  // Build the *initial* overview (unfiltered). Now reads overview_plot
  const handleMotifOverview = async () => {
    if (compareMode) {
      if (!sessionId) return alert("Missing session_id");

      const fd = new FormData();
      fd.append("session_id", sessionId);
      fd.append("window", String(inputWindow));
      // optional thresholds can be added later:
      // fd.append("per_motif_pvals_json", JSON.stringify({}));

      const json = await fetchJSON(
        `${API_BASE}/plot-motif-overview-compare`,
        { method: "POST", body: fd },
        alert
      );

      if (json) {
        setOverviewFiguresByLabel(json.figures || {});
        setFilteredFiguresByLabel({});

        // json.ordered_peaks is { [label]: string[] }
        const map: OrderedPeaksByLabel = {};
        if (json.ordered_peaks && typeof json.ordered_peaks === "object") {
          for (const [label, arr] of Object.entries(json.ordered_peaks)) {
            map[label] = Array.isArray(arr) ? arr.map(String) : [];
          }
        }
        setOrderedPeaksByLabel(map);
      }
      return;
    }

    // --- existing single-list path ---
    if (!dataId) return alert("Missing data_id");
    const fd = new FormData();
    motifs.forEach((m) => {
      const payload = {
        type: m.type,
        data: m.type === "iupac" ? m.iupac : m.type === "pwm" ? m.pwm : m.pcm,
        name: m.name,
        color: m.color,
      };
      fd.append("motifs", JSON.stringify(payload));
    });
    fd.append("window", String(inputWindow));
    fd.append("data_id", dataId);

    const json = await fetchJSON(
      `${API_BASE}/plot-motif-overview`,
      { method: "POST", body: fd },
      alert
    );
    if (json) {
      setOverviewFigureJson(json.overview_plot ?? null);
      setFilteredOverviewFigureJson(null);
    }
  };

  const handleGetGenomicInput = async () => {
    if (dataType === "bed" && !bedFile) return alert("Upload a BED");
    if (dataType === "genes" && !geneListFile)
      return alert("Upload a gene-list CSV");

    const fd = new FormData();
    if (dataType === "bed") fd.append("bed_file", bedFile!);
    if (dataType === "genes") fd.append("gene_list_file", geneListFile!);
    fd.append("window_size", String(inputWindow));

    const json = await fetchJSON(
      `${API_BASE}/get-genomic-input`,
      { method: "POST", body: fd },
      alert
    );
    if (json) {
      setDataId(json.data_id);
      setPeakList(json.peak_list);
      setScanComplete(false);
      setOverviewFigureJson(null);
      setFilteredOverviewFigureJson(null);
    }
  };

  const onFinishedScan = async () => {
    setScanComplete(true);

    if (compareMode) {
      if (!sessionId) return alert("Missing session_id");
      const fd = new FormData();
      fd.append("session_id", sessionId);
      fd.append("window", String(inputWindow));
      fd.append("fimo_threshold", "0.005");

      const ok = await fetchJSON(
        `${API_BASE}/get-motif-hits-batch`,
        { method: "POST", body: fd },
        alert
      );
      if (!ok) return;
      await handleMotifOverview(); // builds both figures
      setScanVersion((v) => v + 1);
      return;
    }

    // single
    await handleMotifOverview();
    setScanVersion((v) => v + 1);
  };

  const fetchFilteredPlot = async (filters: FilterSettings) => {
    const {
      openChromatin,
      bindingPeaks,
      sortByHits,
      sortByScore,
      selectedMotif,
    } = filters;

    if (compareMode) {
      if (!sessionId) return alert("Missing session_id");

      const fd = new FormData();
      fd.append("session_id", sessionId);
      fd.append("window", String(inputWindow));
      fd.append("chip", String(bindingPeaks)); // "true"/"false"
      fd.append("atac", String(openChromatin)); // "true"/"false"
      fd.append("use_hit_number", String(sortByHits));
      fd.append("use_match_score", String(sortByScore));
      fd.append("chosen_motif", selectedMotif || "");
      fd.append(
        "per_motif_pvals_json",
        JSON.stringify(filters.perMotifPvals || {})
      );

      const json = await fetchJSON(
        `${API_BASE}/plot-filtered-overview-compare`,
        { method: "POST", body: fd },
        alert
      );
      if (json) {
        setFilteredFiguresByLabel(json.figures || {}); // <- renders filtered plots
      }
      return;
    }

    if (!dataId) return alert("Missing data_id");

    const fd = new FormData();
    fd.append("data_id", dataId);
    fd.append("window", String(inputWindow));
    fd.append("chip", String(bindingPeaks));
    fd.append("atac", String(openChromatin));
    fd.append("use_hit_number", String(sortByHits));
    fd.append("use_match_score", String(sortByScore));
    fd.append("chosen_motif", selectedMotif);
    fd.append(
      "per_motif_pvals_json",
      JSON.stringify(filters.perMotifPvals || {})
    );

    const res = await fetch(`${API_BASE}/plot-filtered-overview`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) return alert(await res.text());

    const json = await res.json();
    // Backend should return { overview_plot, data_id, peak_list } now
    setFilteredOverviewFigureJson(json.overview_plot ?? null);
  };

  const validatedMotifNames = validatedMotifs
    .map((m) => m.name)
    .filter(Boolean);

  return (
    <div className="container-xxl">
      <Container className="my-4">
        <h2 className="text-center mb-4">Motif Viewer</h2>

        {/* Stepper Nav */}
        <Nav variant="pills" className="justify-content-center mb-4">
          {steps.map((label, idx) => (
            <Nav.Item key={idx}>
              <Nav.Link
                active={activeStep === idx}
                onClick={() => setActiveStep(idx as Step)}
              >
                {idx + 1}. {label}
              </Nav.Link>
            </Nav.Item>
          ))}
        </Nav>

        {/* Wizard Content */}
        <Card className="mb-3">
          <Card.Body>
            {activeStep === 0 && (
              <>
                <h4 className="text-center mb-3 mt-1">
                  Set up genomic input{" "}
                  <InfoTip
                    text="Upload a BED file for a single list, or compare two gene lists by CSV. Window size sets +/- bp around TSS."
                    placement="right"
                    id="genomic-input-info"
                  />
                </h4>

                <CompareInputs
                  compareMode={compareMode}
                  setCompareMode={setCompareMode}
                  labelA={labelA}
                  setLabelA={setLabelA}
                  labelB={labelB}
                  setLabelB={setLabelB}
                  geneListFileA={geneListFileA}
                  setGeneListFileA={setGeneListFileA}
                  geneListFileB={geneListFileB}
                  setGeneListFileB={setGeneListFileB}
                />

                <GenomicInput
                  compareMode={compareMode} // NEW: pass compareMode down
                  dataType={dataType}
                  setDataType={setDataType}
                  bedFile={bedFile}
                  setBedFile={setBedFile}
                  geneListFile={geneListFile}
                  setGeneListFile={setGeneListFile}
                  inputWindow={inputWindow}
                  setInputWindow={setInputWindow}
                  onProcess={async () => {
                    if (compareMode) await handleGetGenomicInputCompare();
                    else await handleGetGenomicInput();
                  }}
                  // Optional simple guard so users don’t click “Process” too soon
                  canProcess={
                    compareMode
                      ? Boolean(
                          labelA && labelB && geneListFileA && geneListFileB
                        )
                      : dataType === "bed"
                      ? Boolean(bedFile)
                      : Boolean(geneListFile)
                  }
                />
              </>
            )}

            {activeStep === 1 && (
              <div className="text-center">
                <h4 className="text-center mb-3 mt-3">
                  Set up motif input{" "}
                  <InfoTip
                    text="Enter all the motifs you're interested in. 
          You can input IUPAC consensus sequences, Position Weight/Frequency Matrices (PWM/PFM) or Position Count Matrices.
          Enter a name and choose a color for each motif."
                    placement="right"
                    id="genomic-input-info"
                  />
                </h4>
                {motifs.length === 0 && (
                  <div className="alert alert-warning">
                    Please add at least one motif to proceed!
                  </div>
                )}

                {motifs.map((m, i) => (
                  <GetMotifInput
                    key={i}
                    motif={m}
                    onChange={(p) => updateMotif(i, p)}
                    onRemove={() => removeMotif(i)}
                  />
                ))}

                <Button
                  variant="outline-primary"
                  onClick={addMotif}
                  className="mt-2"
                >
                  + Add Motif
                </Button>
                <div>
                  <Button
                    variant="primary mt-3"
                    onClick={handleProcessMotifs}
                    disabled={isProcessingMotifs}
                  >
                    {isProcessingMotifs ? (
                      <>
                        <Spinner
                          as="span"
                          animation="border"
                          size="sm"
                          role="status"
                          aria-hidden="true"
                          className="me-2 mt-3"
                        />
                        Processing…
                      </>
                    ) : (
                      "Process Motifs"
                    )}
                  </Button>

                  {validationError && (
                    <div className="alert alert-danger mt-3">
                      {validationError}
                    </div>
                  )}

                  {processedSuccess && (
                    <span className="badge bg-success ms-2 mt-3">
                      Processed!
                    </span>
                  )}
                </div>
              </div>
            )}

            {activeStep === 2 && (
              <ScanFIMO
                // existing
                dataId={dataId}
                inputWindow={inputWindow}
                validMotifs={validatedMotifs}
                selectedStremeMotifs={[]}
                discoveredMotifs={[]}
                fetchJSON={fetchJSON}
                onError={alert}
                scanComplete={scanComplete}
                onScanComplete={() => {
                  onFinishedScan();
                  goNext();
                }}
                // NEW
                compareMode={compareMode}
                sessionId={sessionId}
              />
            )}

            {activeStep === 3 &&
              (compareMode ? (
                <GetFilterDataCompare
                  sessionId={sessionId!}
                  scanComplete={scanComplete}
                  scanVersion={scanVersion}
                  onDone={() => {
                    // You can trigger a re-plot here or just advance.
                    // The step 4 UI lets users generate filtered plots as needed.
                    //goNext();
                  }}
                  onError={alert}
                />
              ) : (
                <GetFilterData
                  dataId={dataId}
                  scanComplete={scanComplete}
                  scanVersion={scanVersion}
                  onFilteredOverview={(figureJson: string | null) => {
                    // (Single mode) You were previously using this to advance; keep it.
                    setFilteredOverviewFigureJson(figureJson);
                    //goNext();
                  }}
                  onError={alert}
                />
              ))}

            {activeStep === 4 && (
              <>
                {compareMode ? (
                  <>
                    <FiltersBar
                      motifList={validatedMotifNames}
                      fimoThreshold={fimoThreshold} // from context
                      onApply={(f: FilterSettings) =>
                        fetchFilteredPlot({
                          openChromatin: f.openChromatin,
                          bindingPeaks: f.bindingPeaks,
                          sortByHits: f.sortByHits,
                          sortByScore: f.sortByScore,
                          selectedMotif: f.selectedMotif,
                          perMotifPvals: f.perMotifPvals,
                        })
                      }
                      applyLabel="📊 Generate Plots"
                    />
                    <MotifOccurrenceCompare
                      figuresByLabel={
                        Object.keys(filteredFiguresByLabel).length
                          ? filteredFiguresByLabel
                          : overviewFiguresByLabel
                      }
                    />
                  </>
                ) : (
                  <MotifOccurencePlot
                    figureJson={
                      filteredOverviewFigureJson ||
                      overviewFigureJson ||
                      undefined
                    }
                    onApplyFilters={fetchFilteredPlot}
                    motifList={validatedMotifNames}
                  />
                )}
              </>
            )}

            {activeStep === 5 &&
              (compareMode ? (
                <BigWigOverlayCompare
                  sessionId={sessionId}
                  inputWindow={inputWindow}
                  apiBase={API_BASE}
                  labels={
                    Object.keys(orderedPeaksByLabel).length
                      ? Object.keys(orderedPeaksByLabel)
                      : [labelA, labelB].filter(Boolean)
                  }
                  peakListsByLabel={orderedPeaksByLabel}
                />
              ) : (
                <BigWigOverlay
                  dataId={dataId}
                  inputWindow={inputWindow}
                  peakList={peakList}
                  apiBase={API_BASE}
                />
              ))}
          </Card.Body>

          {/* Navigation Buttons */}
          <Card.Footer className="d-flex justify-content-between">
            <Button
              variant="secondary"
              disabled={activeStep === 0}
              onClick={goBack}
            >
              Back
            </Button>
            {activeStep < steps.length - 1 && (
              <Button variant="primary" onClick={goNext}>
                Next
              </Button>
            )}
          </Card.Footer>
        </Card>
      </Container>
    </div>
  );
}

export default MotifViewer;

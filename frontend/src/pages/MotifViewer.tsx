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

// ---------- Small helpers ----------
const DEFAULT_FILTERS: FilterSettings = {
  openChromatin: false,
  bindingPeaks: false,
  sortByHits: false,
  sortByScore: false,
  selectedMotif: "",
  bestTranscript: false,
  perMotifPvals: {},
};

const ENDPOINTS = {
  getGenomicSingle: (base: string) => `${base}/get-genomic-input`,
  getGenomicCompare: (base: string) => `${base}/get-genomic-input-compare`,
  // enqueue
  validateSingle: (base: string) => `${base}/validate-motifs`,
  validateCompare: (base: string) => `${base}/validate-motifs-group`,

  overviewSingle: (base: string) => `${base}/plot-motif-overview`, // returns {job_id}
  overviewCompare: (base: string) => `${base}/plot-motif-overview-compare`, // returns {job_id}
  filteredSingle: (base: string) => `${base}/plot-filtered-overview`, // returns {job_id}
  filteredCompare: (base: string) => `${base}/plot-filtered-overview-compare`, // returns {job_id}
  getHitsBatch: (base: string) => `${base}/get-motif-hits-batch`, // returns {job_id}

  // job status
  jobStatus: (base: string, id: string) => `${base}/jobs/${id}`,

  // fetch artifacts
  fetchOverviewSingle: (base: string, dataId: string) =>
    `${base}/plots/overview/${dataId}`,
  fetchOverviewCompare: (base: string, sessionId: string) =>
    `${base}/plots/overview-compare/${sessionId}`,
  fetchFilteredSingle: (base: string, dataId: string) =>
    `${base}/plots/filtered/${dataId}`,
  fetchFilteredCompare: (base: string, sessionId: string) =>
    `${base}/plots/filtered-compare/${sessionId}`,

  // downloads
  dlOverviewSingle: (base: string, dataId: string) =>
    `${base}/download/overview/${dataId}`,
  dlOverviewCompareMerged: (base: string, sessionId: string) =>
    `${base}/download/overview-compare/${sessionId}?merged=true`,
  dlOverviewCompareZip: (base: string, sessionId: string) =>
    `${base}/download/overview-compare/${sessionId}?merged=false`,
  dlFilteredSingle: (base: string, dataId: string) =>
    `${base}/download/filtered/${dataId}`,
  dlFilteredCompareMerged: (base: string, sessionId: string) =>
    `${base}/download/filtered-compare/${sessionId}?merged=true`,
  dlFilteredCompareZip: (base: string, sessionId: string) =>
    `${base}/download/filtered-compare/${sessionId}?merged=false`,
};

async function postJSON(url: string, fd: FormData) {
  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
async function postBlob(url: string, fd: FormData) {
  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}
async function getJSON(url: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForJob(
  jobId: string,
  opts?: { timeoutMs?: number; intervalMs?: number }
) {
  const timeoutMs = opts?.timeoutMs ?? 10 * 60 * 1000; // 10 minutes
  const intervalMs = opts?.intervalMs ?? 1500;
  const start = Date.now();
  while (true) {
    const js = await getJSON(ENDPOINTS.jobStatus(API_BASE, jobId));
    if (js.status === "finished") return js;
    if (js.status === "failed") throw new Error(js.error || "Job failed");
    if (Date.now() - start > timeoutMs)
      throw new Error("Timed out waiting for job");
    await sleep(intervalMs);
  }
}

type Step = 0 | 1 | 2 | 3 | 4 | 5;
const steps = [
  "Genomic Input",
  "Enter Motifs",
  "Scan with FIMO",
  "Set Filters",
  "Occurrence Overview",
  "Track Overlay",
];

function MotifViewer() {
  const [lastFilters, setLastFilters] = useState<FilterSettings | null>(null);

  const [activeStep, setActiveStep] = useState<Step>(() => {
    const saved = sessionStorage.getItem("motifViewer.activeStep");
    return saved !== null ? (parseInt(saved, 10) as Step) : 0;
  });
  const goNext = () =>
    setActiveStep((s) => Math.min(steps.length - 1, s + 1) as Step);
  const goBack = () => setActiveStep((s) => Math.max(0, s - 1) as Step);

  const {
    // core input controls
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

    // motifs
    motifs,
    setMotifs,

    // scan state
    scanComplete,
    setScanComplete,

    // compare context
    isCompare,
    setIsCompare,
    labelA,
    setLabelA,
    labelB,
    setLabelB,
    geneListFileA,
    setGeneListFileA,
    geneListFileB,
    setGeneListFileB,
    dataIdA,
    setDataIdA,
    dataIdB,
    setDataIdB,
    sessionId,
    setSessionId,

    // shared
    fimoThreshold,
    labelsByDataId,
    setLabelsByDataId,
    fetchJSON,
  } = useMotifViewer();

  // Figures
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

  // Ordered peaks (used by overlay compare)
  type OrderedPeaksByLabel = Record<string, string[]>;
  const [orderedPeaksByLabel, setOrderedPeaksByLabel] =
    useState<OrderedPeaksByLabel>({});

  // Persist a scan version (used by children props)
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

  // Motif CRUD
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

  // Genomic input (single/compare)
  const handleGetGenomicInputCompare = async () => {
    if (!geneListFileA || !geneListFileB)
      return alert("Upload both gene lists (A & B)");
    const fd = new FormData();
    fd.append("gene_list_file_a", geneListFileA);
    fd.append("gene_list_file_b", geneListFileB);
    fd.append("label_a", labelA);
    fd.append("label_b", labelB);
    fd.append("window_size", String(inputWindow));

    try {
      const json = await postJSON(ENDPOINTS.getGenomicCompare(API_BASE), fd);
      setSessionId(json.session_id);

      // If API returns label-tagged datasets, prefer that; otherwise assume [0]=A,[1]=B
      const dsA =
        (json.datasets || []).find((d: any) => d.label === labelA) ??
        json.datasets?.[0];
      const dsB =
        (json.datasets || []).find((d: any) => d.label === labelB) ??
        json.datasets?.[1];

      setDataIdA(dsA?.data_id ?? null);
      setDataIdB(dsB?.data_id ?? null);

      // Maintain legacy single defaults to the A-side for components still expecting them
      setDataId(dsA?.data_id ?? null);
      setPeakList(Array.isArray(dsA?.peak_list) ? dsA.peak_list : []);

      setScanComplete(false);

      // map data_id -> label for downstream plots
      const mapping: Record<string, string> = {};
      (json.datasets || []).forEach((ds: any) => {
        if (ds?.data_id && ds?.label) mapping[ds.data_id] = ds.label;
      });
      setLabelsByDataId(mapping);

      // clear plots
      setOverviewFigureJson(null);
      setFilteredOverviewFigureJson(null);
      setOverviewFiguresByLabel({});
      setFilteredFiguresByLabel({});
    } catch (e: any) {
      alert(e.message || String(e));
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

    try {
      const json = await postJSON(ENDPOINTS.getGenomicSingle(API_BASE), fd);
      setDataId(json.data_id);
      setPeakList(json.peak_list);
      setScanComplete(false);
      setOverviewFigureJson(null);
      setFilteredOverviewFigureJson(null);
    } catch (e: any) {
      alert(e.message || String(e));
    }
  };

  // Motif validation
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validatedMotifs, setValidatedMotifs] = useState<typeof motifs>([]);
  const [processedSuccess, setProcessedSuccess] = useState(false);
  const [isProcessingMotifs, setIsProcessingMotifs] = useState(false);

  function appendMotifsToForm(fd: FormData) {
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
  }

  const handleProcessMotifs = async (): Promise<void> => {
    setProcessedSuccess(false);
    setValidationError(null);
    setIsProcessingMotifs(true);
    const fd = new FormData();
    appendMotifsToForm(fd);
    try {
      if (isCompare) {
        if (!sessionId) throw new Error("Missing session_id");
        fd.append("session_id", sessionId);
        await postJSON(ENDPOINTS.validateCompare(API_BASE), fd);
      } else {
        if (!dataId) throw new Error("Missing data_id");
        fd.append("data_id", dataId);
        await postJSON(ENDPOINTS.validateSingle(API_BASE), fd);
      }
      setValidatedMotifs([...motifs]);
      setProcessedSuccess(true);
    } catch (err: any) {
      setValidationError(err.message || "Network error");
    } finally {
      setIsProcessingMotifs(false);
    }
  };

  // Overview (unfiltered)
  const handleMotifOverview = async () => {
    try {
      if (isCompare) {
        if (!sessionId) throw new Error("Missing session_id");
        const fd = new FormData();
        fd.append("session_id", sessionId);
        fd.append("window", String(inputWindow));
        const { job_id } = await postJSON(
          ENDPOINTS.overviewCompare(API_BASE),
          fd
        );
        await waitForJob(job_id);
        const json = await getJSON(
          ENDPOINTS.fetchOverviewCompare(API_BASE, sessionId)
        );
        setOverviewFiguresByLabel(json.figures || {});
        setFilteredFiguresByLabel({});
        const map: Record<string, string[]> = {};
        if (json.ordered_peaks && typeof json.ordered_peaks === "object") {
          for (const [label, arr] of Object.entries(json.ordered_peaks)) {
            map[label] = Array.isArray(arr) ? (arr as any[]).map(String) : [];
          }
        }
        setOrderedPeaksByLabel(map);
        return;
      }
      // single
      if (!dataId) throw new Error("Missing data_id");
      const fd = new FormData();
      appendMotifsToForm(fd);
      fd.append("window", String(inputWindow));
      fd.append("data_id", dataId);
      const { job_id } = await postJSON(ENDPOINTS.overviewSingle(API_BASE), fd);
      await waitForJob(job_id);
      const json = await getJSON(
        ENDPOINTS.fetchOverviewSingle(API_BASE, dataId)
      );
      setOverviewFigureJson(
        json.overview_plot ? JSON.stringify(json.overview_plot) : null
      );
      setFilteredOverviewFigureJson(null);
      setLastFilters((prev) => prev ?? DEFAULT_FILTERS);
    } catch (e: any) {
      alert(e.message || String(e));
    }
  };

  // After scan is finished
  const onFinishedScan = async () => {
    setScanComplete(true);
    try {
      if (isCompare) {
        if (!sessionId) throw new Error("Missing session_id");
        const fd = new FormData();
        fd.append("session_id", sessionId);
        fd.append("window", String(inputWindow));
        fd.append("fimo_threshold", "0.005");
        const { job_id } = await postJSON(ENDPOINTS.getHitsBatch(API_BASE), fd);
        await waitForJob(job_id);
      }
      await handleMotifOverview();
      setScanVersion((v) => v + 1);
    } catch (e: any) {
      alert(e.message || String(e));
    }
  };

  // Build filters → FormData (reused)
  function buildFormDataFromFilters(
    filters: FilterSettings,
    inCompare: boolean
  ): FormData {
    const fd = new FormData();
    fd.append("window", String(inputWindow));
    fd.append("chip", String(!!filters.bindingPeaks));
    fd.append("atac", String(!!filters.openChromatin));
    fd.append("use_hit_number", String(!!filters.sortByHits));
    fd.append("use_match_score", String(!!filters.sortByScore));
    fd.append("chosen_motif", filters.selectedMotif || "");
    fd.append("best_transcript", String(!!filters.bestTranscript));
    fd.append(
      "per_motif_pvals_json",
      JSON.stringify(filters.perMotifPvals || {})
    );
    if (inCompare) {
      if (!sessionId) throw new Error("Missing session_id");
      fd.append("session_id", sessionId);
    } else {
      if (!dataId) throw new Error("Missing data_id");
      fd.append("data_id", dataId);
    }
    return fd;
  }

  // Filtered plot (single/compare unified)
  const fetchFilteredPlot = async (filters: FilterSettings) => {
    setLastFilters(filters);
    const inCompare = isCompare;
    try {
      const fd = buildFormDataFromFilters(filters, inCompare);
      const enqueueUrl = inCompare
        ? ENDPOINTS.filteredCompare(API_BASE)
        : ENDPOINTS.filteredSingle(API_BASE);
      const { job_id } = await postJSON(enqueueUrl, fd);
      await waitForJob(job_id);

      const fetchUrl = inCompare
        ? ENDPOINTS.fetchFilteredCompare(API_BASE, sessionId!)
        : ENDPOINTS.fetchFilteredSingle(API_BASE, dataId!);
      const json = await getJSON(fetchUrl);

      if (inCompare) {
        setFilteredFiguresByLabel(json.figures || {});
      } else {
        // keep it as string if your plot component expects string
        setFilteredOverviewFigureJson(
          json.overview_plot ? JSON.stringify(json.overview_plot) : null
        );
      }
    } catch (e: any) {
      alert(e.message || String(e));
    }
  };

  // Downloads (single/compare unified)
  async function downloadFiltered(merge?: boolean) {
    if (!lastFilters)
      return alert("Apply filters at least once before downloading.");
    const inCompare = isCompare;
    try {
      // enqueue a “download build” if your worker only writes CSV/ZIP when asked
      const fd = buildFormDataFromFilters(lastFilters, inCompare);
      fd.set("download", "true");
      if (inCompare) fd.set("merge", String(!!merge));
      const enqueueUrl = inCompare
        ? ENDPOINTS.filteredCompare(API_BASE)
        : ENDPOINTS.filteredSingle(API_BASE);
      const { job_id } = await postJSON(enqueueUrl, fd);
      await waitForJob(job_id);

      // now fetch the ready-made file
      const dlUrl = inCompare
        ? merge
          ? ENDPOINTS.dlFilteredCompareMerged(API_BASE, sessionId!)
          : ENDPOINTS.dlFilteredCompareZip(API_BASE, sessionId!)
        : ENDPOINTS.dlFilteredSingle(API_BASE, dataId!);

      const res = await fetch(dlUrl);
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = inCompare
        ? merge
          ? `${sessionId ?? "motif_hits"}_merged.csv`
          : `${sessionId ?? "motif_hits"}.zip`
        : `${dataId ?? "motif_hits"}_filtered.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      alert(e.message || String(e));
    }
  }

  const validatedMotifNames = validatedMotifs
    .map((m) => m.name)
    .filter(Boolean);

  // ---------- Render ----------
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

                {/* Compare block now bound to context */}
                <CompareInputs
                  compareMode={isCompare}
                  setCompareMode={setIsCompare}
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
                  compareMode={isCompare}
                  dataType={dataType}
                  setDataType={setDataType}
                  bedFile={bedFile}
                  setBedFile={setBedFile}
                  geneListFile={geneListFile}
                  setGeneListFile={setGeneListFile}
                  inputWindow={inputWindow}
                  setInputWindow={setInputWindow}
                  onProcess={async () => {
                    if (isCompare) await handleGetGenomicInputCompare();
                    else await handleGetGenomicInput();
                  }}
                  canProcess={
                    isCompare
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
                    text={`Enter all the motifs you're interested in. 
You can input IUPAC consensus sequences, PWM/PFM or Position Count Matrices.
Enter a name and choose a color for each motif.`}
                    placement="right"
                    id="motif-input-info"
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
                compareMode={isCompare}
                sessionId={sessionId}
              />
            )}

            {activeStep === 3 &&
              (isCompare ? (
                <GetFilterDataCompare
                  sessionId={sessionId!}
                  scanComplete={scanComplete}
                  scanVersion={scanVersion}
                  onDone={() => {}}
                  onError={alert}
                />
              ) : (
                <GetFilterData
                  dataId={dataId}
                  scanComplete={scanComplete}
                  scanVersion={scanVersion}
                  onFilteredOverview={(figureJson: string | null) => {
                    setFilteredOverviewFigureJson(figureJson);
                  }}
                  onError={alert}
                />
              ))}

            {activeStep === 4 && (
              <>
                {isCompare ? (
                  <>
                    <FiltersBar
                      motifList={validatedMotifNames}
                      fimoThreshold={fimoThreshold}
                      onApply={(f: FilterSettings) =>
                        fetchFilteredPlot({
                          openChromatin: f.openChromatin,
                          bindingPeaks: f.bindingPeaks,
                          sortByHits: f.sortByHits,
                          sortByScore: f.sortByScore,
                          selectedMotif: f.selectedMotif,
                          bestTranscript: f.bestTranscript,
                          perMotifPvals: f.perMotifPvals,
                        })
                      }
                      applyLabel="📊 Generate Plots"
                    />
                    <>
                      <Button
                        variant="outline-success"
                        size="sm"
                        disabled={!sessionId || !lastFilters}
                        onClick={() => downloadFiltered(true)}
                      >
                        Download hits (merged CSV)
                      </Button>
                      <Button
                        variant="outline-secondary"
                        size="sm"
                        disabled={!sessionId || !lastFilters}
                        onClick={() => downloadFiltered(false)}
                      >
                        Download hits (per-dataset ZIP)
                      </Button>
                    </>
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
                  >
                    <Button
                      variant="outline-success"
                      size="sm"
                      disabled={!dataId || !lastFilters}
                      onClick={() => downloadFiltered()}
                    >
                      Download hits (CSV)
                    </Button>
                  </MotifOccurencePlot>
                )}
              </>
            )}

            {activeStep === 5 &&
              (isCompare ? (
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

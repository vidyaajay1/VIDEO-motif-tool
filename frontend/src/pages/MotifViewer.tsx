import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Container, Card, Nav, Spinner, Button } from "react-bootstrap";

import GenomicInput from "../components/GenomicInput";
import GetMotifInput from "../components/GetMotifInput";
import ScanFIMO from "../components/ScanFIMO";
import GetFilterData from "../components/GetFilterData";
import MotifOccurencePlot from "../components/MotifOccurencePlot";
import BigWigOverlay from "../components/BigWigOverlay";
import InfoTip from "../components/InfoTip";

import { useMotifViewer } from "../context/MotifViewerContext";
import type { FilterSettings } from "../components/MotifOccurencePlot";

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
    overviewUrl,
    setOverviewUrl,
    filteredOverviewUrl,
    setFilteredOverviewUrl,
    fetchJSON,
  } = useMotifViewer();

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

  //process motifs
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validatedMotifs, setValidatedMotifs] = useState<typeof motifs>([]);
  const [processedSuccess, setProcessedSuccess] = useState(false);
  const [isProcessingMotifs, setIsProcessingMotifs] = useState(false);

  const handleProcessMotifs = async (): Promise<void> => {
    setProcessedSuccess(false);
    setValidationError(null);
    setIsProcessingMotifs(true);

    // assemble FormData exactly like your /get-motif-hits expects:
    const fd = new FormData();
    motifs.forEach((m) => {
      // build the payload with the correct key for each motif type
      const payload: Record<string, any> = {
        name: m.name,
        color: m.color,
        type: m.type,
      };
      if (m.type === "pwm") {
        payload.pwm = m.pwm; // use the "pwm" key
      } else if (m.type === "pcm") {
        payload.pcm = m.pcm; // use the "pcm" key
      } else {
        payload.iupac = m.iupac;
      }

      // now append the correctly-shaped JSON
      fd.append("motifs", JSON.stringify(payload));
    });

    fd.append("data_id", dataId!);

    try {
      // call your new validate endpoint
      await fetchJSON(
        `${API_BASE}/validate-motifs`,
        { method: "POST", body: fd },
        (detailMsg: string) => {
          // FastAPI will respond { detail: "Invalid IUPAC ..." }
          setValidationError(detailMsg);
        }
      );

      // if we get here, motifs are valid
      setValidatedMotifs([...motifs]);
      setProcessedSuccess(true);
    } catch (err: any) {
      // fetchJSON will throw an Error with message = HTTPException detail
      setValidationError("Network error: " + err.message);
    } finally {
      setIsProcessingMotifs(false);
    }
  };

  const handleMotifOverview = async () => {
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
      console.log(json);
      setOverviewUrl(json.overview_image);
      setFilteredOverviewUrl(null);
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
      console.log(json);
      setDataId(json.data_id);
      setPeakList(json.peak_list);
      setScanComplete(false);
      setOverviewUrl(null);
      setFilteredOverviewUrl(null);
    }
  };
  const onFinishedScan = () => {
    setScanComplete(true);
    handleMotifOverview();
    +setScanVersion((v) => v + 1); // <- bump version
  };

  const fetchFilteredPlot = async (filters: FilterSettings) => {
    const {
      openChromatin,
      bindingPeaks,
      sortByHits,
      sortByScore,
      selectedMotif,
    } = filters;

    if (!dataId) return alert("Missing data_id");

    const fd = new FormData();
    fd.append("data_id", dataId);
    fd.append("window", String(inputWindow));
    fd.append("chip", String(bindingPeaks));
    fd.append("atac", String(openChromatin));
    fd.append("use_hit_number", String(sortByHits)); // ✅
    fd.append("use_match_score", String(sortByScore)); // ✅
    fd.append("chosen_motif", selectedMotif); // ✅

    const res = await fetch(`${API_BASE}/plot-filtered-overview`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) return alert(await res.text());

    const json = await res.json();
    setFilteredOverviewUrl(json.overview_image);
  };
  const validatedMotifNames = validatedMotifs
    .map((m) => m.name)
    .filter(Boolean);

  return (
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
            <GenomicInput
              dataType={dataType}
              setDataType={setDataType}
              bedFile={bedFile}
              setBedFile={setBedFile}
              geneListFile={geneListFile}
              setGeneListFile={setGeneListFile}
              inputWindow={inputWindow}
              setInputWindow={setInputWindow}
              onProcess={async () => {
                // call handleGetGenomicInput then goNext()
                await handleGetGenomicInput();
                //goNext();
              }}
            />
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
                  <span className="badge bg-success ms-2 mt-3">Processed!</span>
                )}
              </div>
            </div>
          )}

          {activeStep === 2 && (
            <ScanFIMO
              dataId={dataId!}
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
            />
          )}

          {activeStep === 3 && (
            <GetFilterData
              dataId={dataId}
              scanComplete={scanComplete}
              scanVersion={scanVersion}
              onFilteredOverview={(url) => {
                setFilteredOverviewUrl(url);
                goNext();
              }}
              onError={alert}
            />
          )}

          {activeStep === 4 && (
            <MotifOccurencePlot
              plotSrc={filteredOverviewUrl || overviewUrl || ""}
              onApplyFilters={fetchFilteredPlot}
              motifList={validatedMotifNames} //
            />
          )}

          {activeStep === 5 && (
            <BigWigOverlay
              dataId={dataId}
              inputWindow={inputWindow}
              peakList={peakList}
              apiBase={API_BASE}
            />
          )}
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
  );
}

export default MotifViewer;

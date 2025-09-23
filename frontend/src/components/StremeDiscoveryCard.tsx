// StremeDiscoveryCard.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Card,
  Form,
  Button,
  Row,
  Col,
  Spinner,
  ProgressBar,
} from "react-bootstrap";

//const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

interface StremeDiscoveryCardProps {
  selectedStage: string;
  selectedTissue: string;
}

type StremeJobResult = {
  motifs: any[];
  streme_html_url: string;
  tmp_id: string;
};

const POLL_MS = 1500;

const StremeDiscoveryCard: React.FC<StremeDiscoveryCardProps> = ({
  selectedStage,
  selectedTissue,
}) => {
  const [inputSource, setInputSource] = useState<"de_genes" | "uploaded">(
    "de_genes"
  );
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [minWidth, setMinWidth] = useState<number>(6);
  const [maxWidth, setMaxWidth] = useState<number>(10);
  const [windowSize, setWindowSize] = useState<number>(500);

  // Job state
  const [jobId, setJobId] = useState<string | null>(null);
  const [tmpId, setTmpId] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<StremeJobResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const pollRef = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => {
    return () => stopPolling(); // cleanup on unmount
  }, []);

  useEffect(() => {
    if (!jobId) return;

    // start polling
    stopPolling();
    pollRef.current = window.setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/jobs/${jobId}`);
        if (!r.ok) throw new Error(await r.text());
        const j = await r.json(); // {status, progress, result?}

        setStatus(j.status ?? null);
        setProgress(typeof j.progress === "number" ? j.progress : 0);

        if (j.status === "finished") {
          stopPolling();
          // normalize shape
          if (j.result && j.result.motifs) {
            setResult(j.result as StremeJobResult);
          } else {
            setResult({ motifs: [], streme_html_url: "", tmp_id: tmpId || "" });
          }
          setIsRunning(false);
          setStatus("finished");
          setJobId(null);
        } else if (j.status === "failed") {
          stopPolling();
          setIsRunning(false);
          setStatus("finished");
          setJobId(null);
          console.error(j.error || "STREME job failed");
          alert("STREME job failed. Check server logs.");
        }
      } catch (e) {
        stopPolling();
        setIsRunning(false);
        console.error(e);
        alert(`Error polling STREME job: ${e}`);
      }
    }, POLL_MS) as unknown as number;
  }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRunStreme = async () => {
    setIsRunning(true);
    setResult(null);
    setJobId(null);
    setTmpId(null);
    setProgress(0);
    setStatus("queued");

    const formData = new FormData();
    formData.append("minw", String(minWidth));
    formData.append("maxw", String(maxWidth));
    formData.append("window_size", String(windowSize));
    formData.append(
      "use_de_genes",
      inputSource === "de_genes" ? "true" : "false"
    );

    if (inputSource === "de_genes") {
      formData.append("tissue", selectedTissue);
      formData.append("stage", selectedStage);
    } else if (uploadedFile) {
      formData.append("gene_file", uploadedFile);
    }

    try {
      const res = await fetch(`${API_BASE}/run-streme`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json(); // { job_id, tmp_id, status, poll_url? }
      setJobId(data.job_id);
      setTmpId(data.tmp_id ?? null);
      setStatus(data.status ?? "queued");
      // polling starts via useEffect
    } catch (err) {
      console.error("Error running STREME:", err);
      alert(`Failed to run STREME: ${err}`);
      setIsRunning(false);
    }
  };

  const disabledRun =
    isRunning || (inputSource === "uploaded" && !uploadedFile);

  return (
    <Card className="p-4 mt-4">
      <h5>Motif Discovery for Genes</h5>
      <div>
        <small className="form-text text-muted mt-10 ms-10">
          STREME: Bailey, L. T. <i>Bioinformatics</i> 2021
        </small>
      </div>

      <Form.Group className="mb-3">
        <Form.Label>Choose Input Source</Form.Label>
        <Form.Check
          type="radio"
          label="Top 250 DE Genes from Selected Tissue (Recommended)"
          checked={inputSource === "de_genes"}
          onChange={() => setInputSource("de_genes")}
          disabled={!selectedStage || !selectedTissue}
        />
        <Form.Check
          type="radio"
          label="Upload Your Own Gene List (CSV)"
          checked={inputSource === "uploaded"}
          onChange={() => setInputSource("uploaded")}
        />
      </Form.Group>

      {inputSource === "uploaded" && (
        <Form.Group className="mb-3">
          <Form.Label>Upload Gene List (CSV with gene names)</Form.Label>
          <Form.Control
            type="file"
            accept=".csv"
            onChange={(e) => {
              const input = e.target as HTMLInputElement;
              setUploadedFile(input.files?.[0] || null);
            }}
          />
        </Form.Group>
      )}

      <Row>
        <Col md={4}>
          <Form.Label>Window (bp) around TSS </Form.Label>
          <Form.Control
            type="number"
            min={4}
            value={windowSize}
            onChange={(e) => setWindowSize(Number(e.target.value))}
          />
        </Col>
        <Col md={4}>
          <Form.Label>Min Motif Width</Form.Label>
          <Form.Control
            type="number"
            min={4}
            value={minWidth}
            onChange={(e) => setMinWidth(Number(e.target.value))}
          />
        </Col>
        <Col md={4}>
          <Form.Label>Max Motif Width</Form.Label>
          <Form.Control
            type="number"
            min={minWidth}
            value={maxWidth}
            onChange={(e) => setMaxWidth(Number(e.target.value))}
          />
        </Col>
      </Row>

      <Button
        className="mt-3"
        variant="primary"
        onClick={handleRunStreme}
        disabled={disabledRun}
      >
        {isRunning || jobId ? (
          <>
            <Spinner animation="border" size="sm" className="me-2" />
            {status === "finished"
              ? "Finalizing..."
              : status === "failed"
              ? "Failed"
              : status === "started"
              ? `Running STREME... ${progress || 0}%`
              : "Queued..."}
          </>
        ) : (
          "Run STREME"
        )}
      </Button>

      {jobId && status && status !== "finished" && (
        <div className="mt-3">
          <ProgressBar now={progress || 0} label={`${progress || 0}%`} />
          <div className="text-muted small mt-1">Job: {jobId}</div>
        </div>
      )}

      {result && (
        <div className="mt-4">
          {Array.isArray(result.motifs) && result.motifs.length > 0 ? (
            <p>Discovered {result.motifs.length} motifs!</p>
          ) : (
            <p>No motifs discovered.</p>
          )}

          {result.streme_html_url && (
            <a
              href={result.streme_html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-outline-primary mt-2"
            >
              View Full STREME Report{" "}
              <i className="bi bi-box-arrow-up-right ms-2"></i>
            </a>
          )}

          {(result.tmp_id || tmpId) && (
            <a
              href={`${API_BASE}/download-streme/${result.tmp_id ?? tmpId}`}
              className="btn btn-outline-success mt-2 ms-3"
            >
              Download Discovered Motifs (MEME Format)
            </a>
          )}
        </div>
      )}
    </Card>
  );
};

export default StremeDiscoveryCard;

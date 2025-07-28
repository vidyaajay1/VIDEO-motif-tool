import React, { useState } from "react";
import { Card, Form, Button, Row, Col, Spinner } from "react-bootstrap";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

interface StremeDiscoveryCardProps {
  selectedStage: string;
  selectedTissue: string;
}

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
  const [result, setResult] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);

  const handleRunStreme = async () => {
    setIsRunning(true);
    setResult(null);

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

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Error running STREME:", err);
      alert(`Failed to run STREME: ${err}`);
    } finally {
      setIsRunning(false);
    }
  };

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
        disabled={isRunning || (inputSource === "uploaded" && !uploadedFile)}
      >
        {isRunning ? (
          <>
            <Spinner animation="border" size="sm" /> Running STREME...
          </>
        ) : (
          "Run STREME"
        )}
      </Button>
      {result && (
        <div className="mt-4">
          {result.motifs.length > 0 ? (
            <p>Discovered {result.motifs.length} motifs!</p>
          ) : (
            <p>No motifs discovered.</p>
          )}

          <a
            href={result.streme_html_url}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-outline-primary mt-2"
          >
            View Full STREME Report{" "}
            <i className="bi bi-box-arrow-up-right ms-2"></i>
          </a>

          <a
            href={`${API_BASE}/download-streme/${result.tmp_id}`}
            className="btn btn-outline-success mt-2 ms-3"
          >
            Download Discovered Motifs (MEME Format)
          </a>
        </div>
      )}
    </Card>
  );
};

export default StremeDiscoveryCard;

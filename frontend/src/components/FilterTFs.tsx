import React, { useState } from "react";
import { Card, Form, Button, Spinner, Alert, Collapse } from "react-bootstrap";
import { useTFContext } from "../context/TFContext";

//const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const API_BASE = import.meta.env.VITE_API_URL ?? "/api";
interface TFResult {
  symbol: string;
  flybase_id: string;
  motif_id: string | null;
  pfm: number[][] | null;
}

const FilterTFs: React.FC = () => {
  const { state } = useTFContext();
  const { tissue, stage } = state;

  const [inputSource, setInputSource] = useState<"de_genes" | "uploaded">(
    "de_genes"
  );
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isFiltering, setIsFiltering] = useState(false);
  const [tfResults, setTfResults] = useState<TFResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const handleFilter = async () => {
    setIsFiltering(true);
    setTfResults([]);
    setError(null);

    try {
      const formData = new FormData();
      formData.append(
        "use_de_genes",
        inputSource === "de_genes" ? "true" : "false"
      );

      if (inputSource === "de_genes") {
        formData.append("tissue", tissue);
        formData.append("stage", stage);
      } else if (uploadedFile) {
        formData.append("gene_file", uploadedFile);
      } else {
        setError("Please upload a gene list first.");
        setIsFiltering(false);
        return;
      }

      const res = await fetch(`${API_BASE}/filter-tfs`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setTfResults(data.tfs ?? []);
    } catch (err: any) {
      console.error(err);
      setError(err.message ?? "Unknown error occurred.");
    } finally {
      setIsFiltering(false);
    }
  };

  return (
    <Card className="p-4 mb-4">
      <h5>Filter Gene List for Transcription Factors</h5>
      <Form>
        <Form.Group className="mb-2">
          <Form.Check
            type="radio"
            label="Use Top 250 DE Gene List for Selected Tissue"
            checked={inputSource === "de_genes"}
            onChange={() => setInputSource("de_genes")}
            disabled={!stage || !tissue}
          />
          <Form.Check
            type="radio"
            label="Upload Your Own Gene List (CSV)"
            checked={inputSource === "uploaded"}
            onChange={() => setInputSource("uploaded")}
          />
        </Form.Group>

        {inputSource === "uploaded" && (
          <Form.Group controlId="formFile" className="mb-3">
            <Form.Control
              type="file"
              accept=".csv"
              onChange={(e) => {
                const input = e.target as HTMLInputElement;
                setUploadedFile(input.files?.[0] ?? null);
              }}
            />
          </Form.Group>
        )}

        <Button
          onClick={handleFilter}
          disabled={
            isFiltering || (inputSource === "uploaded" && !uploadedFile)
          }
        >
          {isFiltering ? (
            <>
              <Spinner
                as="span"
                animation="border"
                size="sm"
                role="status"
                aria-hidden="true"
              />{" "}
              Filtering...
            </>
          ) : (
            "Filter TFs"
          )}
        </Button>
      </Form>

      {error && (
        <Alert variant="danger" className="mt-3">
          {error}
        </Alert>
      )}

      {tfResults.length > 0 && (
        <div className="mt-3">
          <h6>Filtered Transcription Factors</h6>
          <table className="table table-sm">
            <thead>
              <tr>
                <th>Gene Symbol</th>
                <th>FlyBase ID</th>
                <th>Motif ID</th>
                <th>View PFM</th>
              </tr>
            </thead>
            <tbody>
              {tfResults.map((tf, i) => (
                <React.Fragment key={i}>
                  <tr>
                    <td>{tf.symbol}</td>
                    <td>{tf.flybase_id}</td>
                    <td>{tf.motif_id || <em>No motif found</em>}</td>
                    <td>
                      {tf.pfm ? (
                        <Button
                          variant="link"
                          size="sm"
                          onClick={() =>
                            setExpandedIdx(expandedIdx === i ? null : i)
                          }
                        >
                          {expandedIdx === i ? "Hide" : "View"}
                        </Button>
                      ) : (
                        <em>—</em>
                      )}
                    </td>
                  </tr>
                  {tf.pfm && (
                    <tr>
                      <td colSpan={4} style={{ padding: 0 }}>
                        <Collapse in={expandedIdx === i}>
                          <div className="p-3 bg-light">
                            <div className="d-flex flex-column flex-md-row gap-4 align-items-start">
                              {/* PFM Display */}
                              <div className="flex-fill">
                                <strong>Position Frequency Matrix (PFM)</strong>
                                <pre className="mb-0 mt-2">
                                  {tf.pfm
                                    .map((row) =>
                                      row.map((v) => v.toFixed(6)).join("\t")
                                    )
                                    .join("\n")}
                                </pre>
                              </div>

                              {/* Motif Logo Display */}
                              {tf.motif_id && (
                                <div>
                                  <strong>Motif Logo</strong>
                                  <div className="mt-2">
                                    <img
                                      src={`${API_BASE}/static/svg/${tf.motif_id}.svg`}
                                      alt={`Motif logo for ${tf.symbol}`}
                                      style={{
                                        maxWidth: "400px",
                                        maxHeight: "150px",
                                      }}
                                    />
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </Collapse>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>

          <small className="text-muted">
            Showing {tfResults.length} transcription factors found.
          </small>

          <div className="d-flex gap-2 mt-2">
            <Button
              variant="outline-primary"
              onClick={() => {
                const csvContent =
                  "Gene Symbol,FlyBase ID,Motif ID\n" +
                  tfResults
                    .map(
                      (tf) =>
                        `${tf.symbol},${tf.flybase_id},${tf.motif_id ?? ""}`
                    )
                    .join("\n");
                const blob = new Blob([csvContent], {
                  type: "text/csv;charset=utf-8;",
                });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.setAttribute("download", `Filtered_TFs_with_Motifs.csv`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
              }}
            >
              Download TF List (CSV)
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
};

export default FilterTFs;

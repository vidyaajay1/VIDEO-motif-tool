// src/components/ProcessTomtom.tsx
import React, { useState, ChangeEvent } from "react";
import { Card, Form, Button, Table, Spinner } from "react-bootstrap";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export interface MotifGenePair {
  motif: string;
  gene: string;
}

interface ProcessTomtomProps {
  /**
   * Your in-memory Top 250 DE genes list
   * (you can still pass it down here as a prop)
   */
  top250Genes: string[];
}

const ProcessTomtom: React.FC<ProcessTomtomProps> = ({ top250Genes }) => {
  const [file, setFile] = useState<File | null>(null);
  const [pairs, setPairs] = useState<MotifGenePair[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const handleSubmit = async () => {
    if (!file) return;
    setIsRunning(true);
    setPairs([]);

    const formData = new FormData();
    formData.append("tsv_file", file);
    formData.append("gene_list_json", JSON.stringify(top250Genes));

    try {
      const res = await fetch(`${API_BASE}/process-tomtom`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text);
      }
      const json = await res.json();
      setPairs(json.pairs || []);
    } catch (err: any) {
      console.error("Failed to process TOMTOM:", err);
      alert("Error: " + err.message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <Card className="p-4 mt-4">
      <h5>Intersect TOMTOM Results with DE Gene List</h5>
      <Form.Group className="mb-3">
        <Form.Label>Upload TSV</Form.Label>
        <Form.Control
          type="file"
          accept=".tsv,.txt"
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setFile(e.target.files?.[0] ?? null)
          }
        />
      </Form.Group>

      <Button
        onClick={handleSubmit}
        disabled={!file || isRunning}
        variant="primary"
      >
        {isRunning ? (
          <>
            <Spinner animation="border" size="sm" /> Processing…
          </>
        ) : (
          "Run Mapping"
        )}
      </Button>
      <div className="mt-4">
        <h6>Unique (Motif → Gene) pairs — count: {pairs.length}</h6>
        {pairs.length === 0 ? (
          <p>
            <em>No pairs found</em>
          </p>
        ) : (
          <Table striped bordered hover size="sm">
            <thead>
              <tr>
                <th>Motif</th>
                <th>Gene</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((p, idx) => (
                <tr key={idx}>
                  <td>{p.motif}</td>
                  <td>{p.gene}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </div>
    </Card>
  );
};

export default ProcessTomtom;

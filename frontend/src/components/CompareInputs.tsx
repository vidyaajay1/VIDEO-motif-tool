// components/CompareInputs.tsx
import React from "react";
import { Form, Row, Col } from "react-bootstrap";

type Props = {
  compareMode: boolean;
  setCompareMode: (v: boolean) => void;

  labelA: string;
  setLabelA: (s: string) => void;
  labelB: string;
  setLabelB: (s: string) => void;

  geneListFileA: File | null;
  setGeneListFileA: (f: File | null) => void;
  geneListFileB: File | null;
  setGeneListFileB: (f: File | null) => void;
};

export default function CompareInputs(p: Props) {
  const {
    compareMode,
    setCompareMode,
    labelA,
    setLabelA,
    labelB,
    setLabelB,
    geneListFileA,
    setGeneListFileA,
    geneListFileB,
    setGeneListFileB,
  } = p;

  return (
    <>
      <Form.Check
        type="switch"
        id="compare-mode"
        label="Compare two gene lists"
        checked={compareMode}
        onChange={(e) => setCompareMode(e.currentTarget.checked)}
        className="mb-3"
      />
      {compareMode && (
        <Row>
          <Col md={6} className="mb-3">
            <Form.Label>Label (List A)</Form.Label>
            <Form.Control
              value={labelA}
              onChange={(e) => setLabelA(e.target.value)}
            />
            <Form.Label className="mt-2">Gene list CSV (A)</Form.Label>
            <Form.Control
              type="file"
              accept=".csv"
              onChange={(e) =>
                setGeneListFileA(
                  (e.target as HTMLInputElement).files?.[0] ?? null
                )
              }
            />
          </Col>
          <Col md={6} className="mb-3">
            <Form.Label>Label (List B)</Form.Label>
            <Form.Control
              value={labelB}
              onChange={(e) => setLabelB(e.target.value)}
            />
            <Form.Label className="mt-2">Gene list CSV (B)</Form.Label>
            <Form.Control
              type="file"
              accept=".csv"
              onChange={(e) =>
                setGeneListFileB(
                  (e.target as HTMLInputElement).files?.[0] ?? null
                )
              }
            />
          </Col>
        </Row>
      )}
    </>
  );
}

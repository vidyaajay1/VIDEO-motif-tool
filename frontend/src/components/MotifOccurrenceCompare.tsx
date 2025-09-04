// components/MotifOccurrenceCompare.tsx
import React from "react";
import Plot from "react-plotly.js";
import { Row, Col, Alert } from "react-bootstrap";

type Props = {
  figuresByLabel: Record<string, string>; // label -> plotly JSON string
};

export default function MotifOccurrenceCompare({ figuresByLabel }: Props) {
  const labels = Object.keys(figuresByLabel);
  if (labels.length === 0)
    return <Alert variant="warning">No plots yet.</Alert>;

  return (
    <Row>
      {labels.map((label) => {
        const fig = JSON.parse(figuresByLabel[label]);
        return (
          <Col md={6} className="mb-4" key={label}>
            <h5 className="text-center mb-2">{label}</h5>
            <Plot
              data={fig.data}
              layout={fig.layout}
              config={fig.config}
              style={{ width: "100%" }}
            />
          </Col>
        );
      })}
    </Row>
  );
}

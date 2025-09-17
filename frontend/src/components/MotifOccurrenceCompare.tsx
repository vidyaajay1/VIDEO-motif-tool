// components/MotifOccurrenceCompare.tsx
import React from "react";
import Plot from "react-plotly.js";
import { Row, Col, Alert } from "react-bootstrap";

type PlotlyJSON = { data: any[]; layout: any; config?: any };

type Props = {
  // label → Plotly figure object (NOT a JSON string)
  figuresByLabel: Record<string, PlotlyJSON>;
};

export default function MotifOccurrenceCompare({ figuresByLabel }: Props) {
  const labels = Object.keys(figuresByLabel);
  if (labels.length === 0)
    return <Alert variant="warning">No plots yet. Click Generate Plot!</Alert>;

  return (
    <Row>
      {labels.map((label) => {
        const fig = figuresByLabel[label]; // already an object
        return (
          <Col md={6} className="mb-4" key={label}>
            <h5 className="text-center mb-2">{label}</h5>
            <Plot
              data={fig.data}
              layout={fig.layout}
              config={fig.config}
              style={{ width: "100%" }}
              useResizeHandler
            />
          </Col>
        );
      })}
    </Row>
  );
}

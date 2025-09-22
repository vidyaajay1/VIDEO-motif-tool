import React, { useEffect } from "react";
import { Container, Row, Col, Form, Button, Card } from "react-bootstrap";
import StremeDiscoveryCard from "../components/StremeDiscoveryCard";
import { useTFContext } from "../context/TFContext";
import ProcessTomtom from "../components/ProcessTomtom";
import FilterTFs from "../components/FilterTFs";

// const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const API_BASE = import.meta.env.VITE_API_URL ?? "/api";
const TFFinder: React.FC = () => {
  const { state, dispatch } = useTFContext();
  const { dataSource, stage, tissue, tissueOptions, genes } = state;

  useEffect(() => {
    if (stage) {
      fetch(`${API_BASE}/get-tissues?data_source=${dataSource}&stage=${stage}`)
        .then((res) => {
          if (!res.ok) throw new Error("Failed to fetch tissues");
          return res.json();
        })
        .then((data) =>
          dispatch({ type: "SET_TISSUE_OPTIONS", payload: data.tissues })
        )
        .catch((err) => {
          console.error(err);
          dispatch({ type: "SET_TISSUE_OPTIONS", payload: [] });
        });
    } else {
      dispatch({ type: "SET_TISSUE_OPTIONS", payload: [] });
    }
  }, [stage, dataSource, dispatch]);
  const top250Genes = genes.slice(0, 150).map((g) => g.gene);
  return (
    <Container className="mt-4">
      <h2 className="text-center mb-4">TF Finder</h2>

      <Card className="p-4 mb-4">
        <Row className="mb-3">
          <Col md={4}>
            <Form.Label>Choose Data Source</Form.Label>
            <Form.Select
              value={dataSource}
              onChange={(e) =>
                dispatch({ type: "SET_DATASOURCE", payload: e.target.value })
              }
            >
              <option value="peng2024">Peng et al. 2024 (scRNA-seq)</option>
            </Form.Select>
          </Col>

          <Col md={4}>
            <Form.Label>Developmental Stage</Form.Label>
            <Form.Select
              value={stage}
              onChange={(e) =>
                dispatch({ type: "SET_STAGE", payload: e.target.value })
              }
            >
              <option value="">Select stage</option>
              <option value="10-12">10–12</option>
              <option value="13-16">13–16</option>
            </Form.Select>
          </Col>

          <Col md={4}>
            <Form.Label>Tissue Type</Form.Label>
            <Form.Select
              value={tissue}
              onChange={(e) =>
                dispatch({ type: "SET_TISSUE", payload: e.target.value })
              }
              disabled={!tissueOptions.length}
            >
              <option value="">Select tissue</option>
              {tissueOptions.map((t, i) => (
                <option key={i} value={t}>
                  {t}
                </option>
              ))}
            </Form.Select>
          </Col>
        </Row>

        <Button
          variant="primary"
          disabled={!stage || !tissue}
          onClick={() => {
            fetch(
              `${API_BASE}/get-de-genes?data_source=${dataSource}&stage=${stage}&tissue=${encodeURIComponent(
                tissue
              )}`
            )
              .then((res) => {
                if (!res.ok) throw new Error("Failed to fetch genes");
                return res.json();
              })
              .then((data) =>
                dispatch({ type: "SET_GENES", payload: data.genes })
              )
              .catch((err) => {
                console.error("Error fetching genes:", err);
                dispatch({ type: "SET_GENES", payload: [] });
              });
          }}
        >
          Get DE Genes
        </Button>

        {genes.length > 0 && (
          <Card className="p-4 mt-3">
            <h5>
              Top DE Genes for {tissue} ({stage})
            </h5>
            <ul>
              {genes.slice(0, 10).map((g, i) => (
                <li key={i}>{g.gene}</li>
              ))}
            </ul>
            <small className="text-muted">
              Showing top 10 genes. Download full list for all results.
            </small>
            <div className="d-flex flex-wrap gap-2 mt-3">
              <Button
                variant="outline-primary"
                onClick={() => {
                  const topGenes = genes.slice(0, 250);
                  const csvContent =
                    "Gene,Avg_Log2FC\n" +
                    topGenes.map((g) => `${g.gene},${g.avg_log2FC}`).join("\n");

                  const blob = new Blob([csvContent], {
                    type: "text/csv;charset=utf-8;",
                  });
                  const url = URL.createObjectURL(blob);
                  const link = document.createElement("a");
                  link.href = url;
                  link.setAttribute(
                    "download",
                    `Top_250_Genes_${tissue}_${stage}.csv`
                  );
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);
                }}
              >
                Download Top 250 Genes (+ log2FC)
              </Button>

              <Button
                variant="outline-success"
                onClick={() => {
                  const url = `${API_BASE}/download-de-genes?data_source=${dataSource}&stage=${stage}&tissue=${encodeURIComponent(
                    tissue
                  )}`;
                  window.open(url, "_blank");
                }}
              >
                Download Full Gene List (CSV)
              </Button>
            </div>
          </Card>
        )}
      </Card>
      <FilterTFs />
      <StremeDiscoveryCard selectedStage={stage} selectedTissue={tissue} />
      <div className="mt-4">
        <small className="text">
          We recommend that you run the motif discovery results through a motif
          comparison tool like{" "}
          <a
            href="https://meme-suite.org/meme/tools/tomtom"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline"
          >
            Tomtom
          </a>{" "}
          (Gupta et al. <i>Genome Biology</i> 2007). Here's a helper that can
          give you the intersection of the TOMTOM-reported TFs and the DE gene
          list.
        </small>
        <ProcessTomtom top250Genes={top250Genes} />
      </div>
    </Container>
  );
};

export default TFFinder;

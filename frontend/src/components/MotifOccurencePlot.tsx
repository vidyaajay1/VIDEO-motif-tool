import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Row,
  Col,
  Card,
  Form,
  Accordion,
  ListGroup,
  Button,
} from "react-bootstrap";
import InfoTip from "./InfoTip";

export interface MotifOccurencePlotProps {
  plotSrc: string;
  onApplyFilters: (filters: FilterSettings) => void;
  motifList?: string[]; // Optional: pass motifs from parent
}

export interface FilterSettings {
  openChromatin: boolean;
  bindingPeaks: boolean;
  sortByHits: boolean;
  sortByScore: boolean;
  selectedMotif: string;
}

export default function MotifOccurencePlot({
  plotSrc,
  onApplyFilters,
  motifList = [], // example
}: MotifOccurencePlotProps) {
  const [openChromatin, setOpenChromatin] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [bindingPeaks, setBindingPeaks] = useState(false);
  const [sortByHits, setSortByHits] = useState(false);
  const [sortByScore, setSortByScore] = useState(false);
  const [selectedMotif, setSelectedMotif] = useState("");
  const handleApplyFilters = async () => {
    setIsLoading(true);
    try {
      await onApplyFilters({
        openChromatin,
        bindingPeaks,
        sortByHits,
        sortByScore,
        selectedMotif,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="mb-5 text-center">
      <h4 className="text-center mb-3 mt-3">
        Motif Occurence Overview{" "}
        <InfoTip
          text="Here you'll find a plot of all the motif hits. Toggle the filters and click generate plot to apply the filters 
          (intersection of the input BED files)."
          placement="right"
          id="genomic-input-info"
        />
      </h4>

      <Row className="mt-4">
        {/* Left: Plot */}
        <Col md={9}>
          <Accordion defaultActiveKey="">
            <Accordion.Item eventKey="0">
              <Accordion.Header>Motif Occurrence Plot</Accordion.Header>
              <Accordion.Body
                className="d-flex justify-content-center align-items-center"
                style={{ minHeight: "300px" }}
              >
                {plotSrc ? (
                  <img
                    src={plotSrc}
                    alt="Motif Occurrence"
                    className="img-fluid"
                    style={{ maxHeight: "100%", maxWidth: "100%" }}
                  />
                ) : (
                  <div>No plot generated yet.</div>
                )}
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </Col>

        {/* Right: Settings */}
        <Col md={3}>
          <Card>
            <Card.Header>Plot Settings</Card.Header>
            <Card.Body>
              <Form>
                <strong>Show only:</strong>
                <Form.Group controlId="filterOpenChromatin" className="mb-2">
                  <Form.Check
                    type="checkbox"
                    label="Open chromatin (ATAC) regions"
                    checked={openChromatin}
                    onChange={(e) => setOpenChromatin(e.target.checked)}
                  />
                </Form.Group>
                <Form.Group controlId="filterBindingPeaks" className="mb-3">
                  <Form.Check
                    type="checkbox"
                    label="Binding (ChIP) regions"
                    checked={bindingPeaks}
                    onChange={(e) => setBindingPeaks(e.target.checked)}
                  />
                </Form.Group>
                {/* Sort by motif section */}
                <Form.Group controlId="sortByMotifDropdown" className="mb-3">
                  <Form.Label>
                    <strong>Sort by motif:</strong>
                  </Form.Label>
                  <Form.Select
                    value={selectedMotif}
                    onChange={(e) => setSelectedMotif(e.target.value)}
                  >
                    <option value="">-- Select a motif --</option>
                    {motifList.map((motif) => (
                      <option key={motif} value={motif}>
                        {motif}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>

                {/* Show sorting options only when a motif is selected */}
                {selectedMotif && (
                  <>
                    <Form.Group controlId="sortByHits" className="mb-2">
                      <Form.Check
                        type="checkbox"
                        label="Number of hits"
                        checked={sortByHits}
                        onChange={(e) => setSortByHits(e.target.checked)}
                      />
                    </Form.Group>
                    <Form.Group controlId="sortByScore" className="mb-3">
                      <Form.Check
                        type="checkbox"
                        label="Motif match score"
                        checked={sortByScore}
                        onChange={(e) => setSortByScore(e.target.checked)}
                      />
                    </Form.Group>
                  </>
                )}
                <div className="d-grid gap-2">
                  <Button
                    variant="primary"
                    onClick={handleApplyFilters}
                    disabled={isLoading}
                  >
                    {isLoading ? "Generating…" : "📊 Generate Plot"}
                  </Button>
                </div>
              </Form>
            </Card.Body>
          </Card>

          {plotSrc && (
            <div className="d-grid gap-2 mt-4">
              <a
                href={plotSrc}
                download="motif_occurrence.png"
                className="btn btn-outline-success"
              >
                ⬇ Download Plot
              </a>
            </div>
          )}
        </Col>
      </Row>
    </section>
  );
}

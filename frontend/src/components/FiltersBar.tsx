// components/FiltersBar.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, Form, Accordion, Button } from "react-bootstrap";
import InfoTip from "./InfoTip";

export type FilterSettings = {
  openChromatin: boolean;
  bindingPeaks: boolean;
  sortByHits: boolean;
  sortByScore: boolean;
  selectedMotif: string;
  perMotifPvals: Record<string, number>;
};

type Props = {
  motifList: string[];
  fimoThreshold: string; // from context
  onApply: (filters: FilterSettings) => void | Promise<void>;
  applyLabel?: string; // default: "📊 Generate Plot"
};

export default function FiltersBar({
  motifList,
  fimoThreshold,
  onApply,
  applyLabel = "📊 Generate Plot",
}: Props) {
  const [openChromatin, setOpenChromatin] = useState(false);
  const [bindingPeaks, setBindingPeaks] = useState(false);
  const [sortByHits, setSortByHits] = useState(false);
  const [sortByScore, setSortByScore] = useState(false);
  const [selectedMotif, setSelectedMotif] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [perMotifPvals, setPerMotifPvals] = useState<Record<string, string>>(
    {}
  );

  // keep threshold map in sync with available motifs
  useEffect(() => {
    setPerMotifPvals((prev) => {
      const next = { ...prev };
      motifList.forEach((m) => {
        if (!(m in next)) next[m] = "";
      });
      Object.keys(next).forEach((k) => {
        if (!motifList.includes(k)) delete next[k];
      });
      return next;
    });
  }, [motifList]);

  const setOnePval = (motif: string, value: string) =>
    setPerMotifPvals((prev) => ({ ...prev, [motif]: value }));

  // p-value slider helpers
  const MIN_P = 1e-10;
  const { pToSlider, sliderToP, rightTick } = useMemo(() => {
    const maxPraw = Number(fimoThreshold);
    const maxP =
      Number.isFinite(maxPraw) && maxPraw > 0 && maxPraw <= 1 ? maxPraw : 1;
    const Smax = 10;
    const Smin = Math.max(0, -Math.log10(Math.max(maxP, MIN_P)));
    const span = Math.max(1e-9, Smax - Smin);

    const pToSlider = (pStr?: string) => {
      const p = Number(pStr);
      if (!pStr || Number.isNaN(p) || p <= 0) {
        const sDef = 3; // 1e-3
        return ((Smax - sDef) / span) * 10;
      }
      const s = Math.min(Smax, Math.max(Smin, -Math.log10(Math.max(p, MIN_P))));
      return ((Smax - s) / span) * 10;
    };

    const sliderToP = (r: number) => {
      const s = Smax - (r / 10) * span;
      const p = Math.pow(10, -s);
      return p.toExponential(2);
    };

    return {
      pToSlider,
      sliderToP,
      rightTick: maxP === 1 ? "1" : maxP.toExponential(1),
    };
  }, [fimoThreshold]);

  const handleApply = async () => {
    setIsLoading(true);
    try {
      const clean: Record<string, number> = {};
      Object.entries(perMotifPvals).forEach(([k, v]) => {
        if (!v) return;
        const num = Number(v);
        if (!Number.isNaN(num) && num >= 0) clean[k] = num;
      });
      await onApply({
        openChromatin,
        bindingPeaks,
        sortByHits,
        sortByScore,
        selectedMotif,
        perMotifPvals: clean,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="mb-3">
      <Card.Header>Plot Settings</Card.Header>
      <Card.Body>
        <Form>
          <strong>Show only:</strong>
          <Form.Group className="mb-2">
            <Form.Check
              type="checkbox"
              label="Open chromatin (ATAC) regions"
              checked={openChromatin}
              onChange={(e) => setOpenChromatin(e.target.checked)}
            />
          </Form.Group>
          <Form.Group className="mb-3">
            <Form.Check
              type="checkbox"
              label="Binding (ChIP) regions"
              checked={bindingPeaks}
              onChange={(e) => setBindingPeaks(e.target.checked)}
            />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>
              <strong>Sort by motif:</strong>
            </Form.Label>
            <Form.Select
              value={selectedMotif}
              onChange={(e) => setSelectedMotif(e.target.value)}
            >
              <option value="">-- Select a motif --</option>
              {motifList.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </Form.Select>
          </Form.Group>

          {selectedMotif && (
            <>
              <Form.Group className="mb-2">
                <Form.Check
                  type="checkbox"
                  label="Number of hits"
                  checked={sortByHits}
                  onChange={(e) => setSortByHits(e.target.checked)}
                />
              </Form.Group>
              <Form.Group className="mb-3">
                <Form.Check
                  type="checkbox"
                  label="Motif match score"
                  checked={sortByScore}
                  onChange={(e) => setSortByScore(e.target.checked)}
                />
              </Form.Group>
            </>
          )}

          {/* Per-motif p-value thresholds */}
          <div style={{ width: "clamp(360px, 60%, 600px)" }}>
            <Accordion className="mb-3">
              <Accordion.Item eventKey="pv">
                <Accordion.Header>Motif p-value thresholds</Accordion.Header>
                <Accordion.Body>
                  {motifList.length === 0 ? (
                    <div className="text-muted">No motifs added.</div>
                  ) : (
                    motifList.map((m) => {
                      const pLabel =
                        perMotifPvals[m] &&
                        !Number.isNaN(Number(perMotifPvals[m]))
                          ? Number(perMotifPvals[m]).toExponential(2)
                          : "none";
                      return (
                        <Form.Group key={m} className="mb-3">
                          <div className="fw-semibold mb-1">{m}</div>
                          <div
                            className="mx-auto"
                            style={{ width: "min(480px, 90%)" }}
                          >
                            <div className="d-flex justify-content-end mb-1">
                              <small className="text-muted">
                                {pLabel === "none"
                                  ? `p ≤ ${rightTick}`
                                  : `p ≤ ${pLabel}`}
                              </small>
                            </div>
                            <Form.Range
                              min={0}
                              max={10}
                              step={0.1}
                              value={pToSlider(perMotifPvals[m])}
                              onChange={(e) => {
                                const r = Number(e.target.value);
                                setOnePval(m, sliderToP(r));
                              }}
                            />
                            <div className="d-flex justify-content-between mt-1">
                              <small>1e-10</small>
                              <small>{rightTick}</small>
                            </div>
                          </div>
                        </Form.Group>
                      );
                    })
                  )}

                  {motifList.length > 0 && (
                    <div className="d-flex gap-1 mt-4">
                      <Button
                        size="sm"
                        variant="outline-secondary"
                        onClick={() =>
                          setPerMotifPvals(
                            Object.fromEntries(motifList.map((m) => [m, ""]))
                          )
                        }
                      >
                        Clear all
                      </Button>
                    </div>
                  )}
                </Accordion.Body>
              </Accordion.Item>
            </Accordion>
          </div>

          <div className="d-flex gap-2">
            <Button
              variant="primary"
              onClick={handleApply}
              disabled={isLoading}
            >
              {isLoading ? "Generating…" : applyLabel}
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
}

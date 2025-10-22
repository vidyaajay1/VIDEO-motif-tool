// pages/Tutorial.tsx
import React, { useMemo } from "react";
import {
  Accordion,
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Container,
  Image,
  ListGroup,
  Row,
} from "react-bootstrap";

// Optional: if you're using react-router
// import { Link } from "react-router-dom";

type Step = {
  title: string;
  text: React.ReactNode;
  img?: string; // public/ path, e.g., "/tutorial/fig-2a_sidebar.png"
  ctaHref?: string; // route to open the relevant page
  ctaLabel?: string;
};

type Section = {
  id: string; // anchor id for TOC
  title: string;
  figureHint?: string; // e.g., "Fig 2A–2D"
  steps: Step[];
};

const Tutorial: React.FC = () => {
  // Centralized content so you can edit copy/screenshots later
  const sections: Section[] = useMemo(
    () => [
      {
        id: "getting-started",
        title: "Getting Started",
        figureHint: "Fig 2A",
        steps: [
          {
            title: "Navigate with the Sidebar",
            text: (
              <>
                Open the collapsible sidebar to access{" "}
                <Badge bg="secondary">Motif Viewer</Badge> and{" "}
                <Badge bg="secondary">TF Finder</Badge>. The{" "}
                <strong>Motif Viewer</strong> generates motif occurrence plots,
                while <strong>TF Finder</strong> helps you assemble starting
                gene lists and discover TFs/motifs of interest.
              </>
            ),
            img: "/tutorial/fig-2a_sidebar.png",
            ctaHref: "/motif-viewer",
            ctaLabel: "Open Motif Viewer",
          },
        ],
      },
      {
        id: "tf-finder",
        title: "Step 1 — Using TF Finder to Acquire Gene & Motif Data",
        figureHint: "Fig 2B-2D",
        steps: [
          {
            title: "Identify Gene Sets of Interest",
            text: (
              <>
                Choose a <em>developmental stage</em> and <em>tissue</em> to
                fetch curated DE gene lists (e.g., Top 250 DE genes from Peng
                et&nbsp;al., 2024). You can also upload your own ranked list
                (recommended if you have p-values or log fold changes).
              </>
            ),
            img: "/tutorial/fig-2b_stage_tissue.png",
            ctaHref: "/tf-finder",
            ctaLabel: "Open TF Finder",
          },
          {
            title: "Filter the List for Transcription Factors",
            text: (
              <>
                Use the <strong>Filter for TFs (FlyBase)</strong> helper to
                extract TFs present in your list. VIDEO shows known motifs (if
                available) with JASPAR ID, PWM, and logo.
              </>
            ),
            img: "/tutorial/fig-2c_tf_filter.png",
            ctaHref: "/tf-finder",
            ctaLabel: "Filter for TFs",
          },
          {
            title: "Discover Motifs (If None Reported)",
            text: (
              <>
                Run <strong>Motif Discovery for Genes</strong> (STREME). Select
                your gene list, optionally set promoter window and motif width
                (defaults to 6–10&nbsp;bp), then click <em>Run STREME</em>. Keep
                selected motifs in PWM (MEME) format to paste directly into
                Motif Viewer later.
              </>
            ),
            img: "/tutorial/fig-2d_streme.png",
            ctaHref: "/tf-finder",
            ctaLabel: "Run STREME",
          },
        ],
      },
      {
        id: "motif-viewer-setup",
        title: "Step 2 — Using Motif Viewer to Create Motif Hit Plots",
        figureHint: "Fig 3A-3D",
        steps: [
          {
            title: "Set Up Genomic Input",
            text: (
              <>
                Upload a <strong>gene list</strong> (regular mode) or{" "}
                <strong>two gene lists</strong> (compare mode). Choose promoter
                window size around the TSS (default ±500&nbsp;bp). Click{" "}
                <em>Process Genomic Input</em> to validate and fetch
                annotations/sequences.
              </>
            ),
            img: "/tutorial/fig-3a_genomic_input.png",
            ctaHref: "/motif-viewer",
            ctaLabel: "Process Genomic Input",
          },
          {
            title: "Provide Motif Input",
            text: (
              <>
                Add motifs as <strong>IUPAC</strong> or matrices{" "}
                <strong>(PWM/PFM/PCM)</strong> by pasting or uploading a .txt.
                Give each motif a name and color, then click{" "}
                <em>Process Motifs</em>.
              </>
            ),
            img: "/tutorial/fig-3b_motif_input.png",
            ctaHref: "/motif-viewer",
            ctaLabel: "Process Motifs",
          },
          {
            title: "Scan for Motif Occurrences",
            text: (
              <>
                Set the global p-value threshold (default <code>0.001</code>)
                and click <em>Scan FIMO</em> to find motif hits across
                promoters.
              </>
            ),
            img: "/tutorial/fig-3c_fimo.png",
            ctaHref: "/motif-viewer",
            ctaLabel: "Scan FIMO",
          },
          {
            title: "Integrate ATAC-seq and ChIP-seq Peaks (Optional)",
            text: (
              <>
                Upload <strong>BED</strong> peak files for chromatin
                accessibility and/or TF binding. Click{" "}
                <em>Filter Motif Hits</em> to enable downstream toggles for
                “open chromatin only” and “ChIP peak only.”
              </>
            ),
            img: "/tutorial/fig-3d_atac_chip.png",
            ctaHref: "/motif-viewer",
            ctaLabel: "Filter Motif Hits",
          },
        ],
      },
      {
        id: "plot-settings",
        title: "Adjusting Plot Settings & Export",
        figureHint: "Fig 4",
        steps: [
          {
            title: "Refine What You See",
            text: (
              <>
                Toggle <em>ATAC/ChIP filters</em>, choose a motif under{" "}
                <strong>Sort by motif</strong>, then sort by{" "}
                <em>Number of Hits</em> or <em>Motif Match Score</em>.
                Optionally show only best transcript per gene for the selected
                motif.
              </>
            ),
            img: "/tutorial/fig-4_settings.png",
            ctaHref: "/motif-viewer",
            ctaLabel: "Open Plot Settings",
          },
          {
            title: "Per-Motif Stringency",
            text: (
              <>
                Under <strong>Motif p-value thresholds</strong>, fine-tune
                individual motif thresholds with sliders to tighten or relax
                matches.
              </>
            ),
          },
          {
            title: "Update & Download",
            text: (
              <>
                Click <em>Generate Plot</em> to apply settings. Download all
                displayed motif hits via <strong>Download merged hits</strong>.
                In compare mode, use <strong>Download individual hits</strong>{" "}
                for per-list archives.
              </>
            ),
          },
        ],
      },
      {
        id: "interpreting-plot",
        title: "Interpreting the Motif Occurrence Plot",
        steps: [
          {
            title: "Reading the Visual",
            text: (
              <>
                Rows = <strong>GeneSymbol_TranscriptID</strong> promoter
                windows, aligned at <strong>TSS = 0</strong> and oriented so
                upstream is left and downstream is right. Darker marks =
                stronger matches. Hover to see matched sequence, coordinates,
                strand, p-value, and score. Use the built-in toolbar to zoom,
                reset, and download the figure.
              </>
            ),
            img: "/tutorial/fig-4_plot_hover.png",
          },
        ],
      },
      {
        id: "signal-overlay",
        title: "Optional — Visualizing ATAC/ChIP BigWig Tracks with Motif Hits",
        figureHint: "Supp. Fig S2",
        steps: [
          {
            title: "Overlay Signal Tracks",
            text: (
              <>
                Upload up to <strong>three BigWig</strong> tracks, name and
                color them, then <em>Process Tracks</em>. Pick a gene under{" "}
                <strong>Select a peak</strong>, choose tracks, and click{" "}
                <em>Generate Overlay</em> to view signals aligned to TSS under
                the motif hits. Y-axes are auto-scaled across tracks; hover to
                read values and coordinates.
              </>
            ),
            img: "/tutorial/fig-s2_overlay.png",
            ctaHref: "/motif-viewer",
            ctaLabel: "Open Overlay View",
          },
        ],
      },
      {
        id: "wrap-up",
        title: "Wrap-Up & Next Steps",
        steps: [
          {
            title: "You're Ready!",
            text: (
              <>
                You can now (1) build motif plots from DE gene sets, (2) compare
                conditions, (3) refine with ATAC/ChIP filters, and (4) overlay
                BigWig signals. Continue exploring with your own data or curated
                sets.
              </>
            ),
          },
        ],
      },
    ],
    []
  );

  return (
    <Container className="my-5">
      <Row className="g-4">
        {/* TOC */}
        <Col lg={3}>
          <Card className="sticky-top" style={{ top: "84px" }}>
            <Card.Header>
              <strong>Tutorial</strong>
            </Card.Header>
            <ListGroup variant="flush">
              {sections.map((s) => (
                <ListGroup.Item key={s.id}>
                  <a
                    href={`#${s.id}`}
                    className="text-decoration-none"
                    onClick={(e) => {
                      // Smooth scroll
                      e.preventDefault();
                      const el = document.getElementById(s.id);
                      if (el) el.scrollIntoView({ behavior: "smooth" });
                    }}
                  >
                    {s.title}
                  </a>
                  {s.figureHint && (
                    <div className="small text-muted mt-1">
                      <em>{s.figureHint}</em>
                    </div>
                  )}
                </ListGroup.Item>
              ))}
            </ListGroup>
          </Card>
        </Col>

        {/* Content */}
        <Col lg={9}>
          <div className="mb-4">
            <h2 className="mb-2">Tutorial: Walkthrough Guide</h2>
            <p className="text-muted mb-0">
              Follow these steps to go from gene lists to motif discovery,
              motif-hit plots, and optional ATAC/ChIP overlays.
            </p>
          </div>

          <Alert variant="info">
            <div className="fw-semibold mb-1">Tip</div>
            You can replace placeholder screenshots in{" "}
            <code>/public/tutorial/</code> with your own images. All buttons
            link to in-app routes — update paths if your router uses different
            URLs.
          </Alert>

          <Accordion alwaysOpen defaultActiveKey={["0"]}>
            {sections.map((section, sIdx) => (
              <Accordion.Item eventKey={String(sIdx)} key={section.id}>
                <Accordion.Header id={section.id}>
                  <div>
                    <div className="fw-semibold">{section.title}</div>
                    {section.figureHint && (
                      <div className="small text-muted">
                        {section.figureHint}
                      </div>
                    )}
                  </div>
                </Accordion.Header>
                <Accordion.Body>
                  {section.steps.map((step, i) => (
                    <Card className="mb-3" key={`${section.id}-${i}`}>
                      <Card.Body>
                        <div className="d-flex justify-content-between align-items-start gap-3">
                          <div style={{ flex: 1 }}>
                            <div className="d-flex align-items-center gap-2 mb-2">
                              <Badge bg="dark">
                                {sIdx + 1}.{i + 1}
                              </Badge>
                              <strong>{step.title}</strong>
                            </div>
                            <div className="mb-3">{step.text}</div>
                            {step.ctaHref && (
                              <>
                                {/* If using react-router, swap Button href+target for Link */}
                                {/* <Button as={Link} to={step.ctaHref} variant="primary"> */}
                                <Button
                                  variant="primary"
                                  href={step.ctaHref}
                                  // target="_self"
                                >
                                  {step.ctaLabel ?? "Open"}
                                </Button>
                              </>
                            )}
                          </div>
                          {step.img && (
                            <div style={{ width: 280, flex: "0 0 280px" }}>
                              <Image
                                src={step.img}
                                alt={step.title}
                                fluid
                                rounded
                                style={{ border: "1px solid #eee" }}
                              />
                              <div className="small text-muted mt-1">
                                Placeholder screenshot
                              </div>
                            </div>
                          )}
                        </div>
                      </Card.Body>
                    </Card>
                  ))}
                </Accordion.Body>
              </Accordion.Item>
            ))}
          </Accordion>
        </Col>
      </Row>
    </Container>
  );
};

export default Tutorial;

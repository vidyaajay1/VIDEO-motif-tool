// pages/Tutorial.tsx
import React, { useMemo } from "react";
import {
  Badge,
  Button,
  Col,
  Container,
  Image,
  ListGroup,
  Row,
} from "react-bootstrap";

type Step = {
  title: string;
  text: React.ReactNode;
  img?: string; // single image (backward compatibility)
  imgs?: string[]; // NEW: multiple images
  ctaHref?: string;
  ctaLabel?: string;
};

type Section = {
  id: string;
  title: string;
  figureHint?: string;
  steps: Step[];
};

const Tutorial: React.FC = () => {
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
            img: "/images/tutorial/fig-2a_sidebar.png",
          },
        ],
      },
      {
        id: "tf-finder",
        title: "Using TF Finder to Acquire Gene & Motif Data",
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
            img: "/images/tutorial/fig-2b_stage_tissue.png",
          },
          {
            title: "Filter the List for Transcription Factors",
            text: (
              <>
                Use the <strong>Filter for TFs</strong> helper to extract TFs
                present in your list, or choose the selected DE genes list.
                VIDEO shows known motifs (if available) with JASPAR ID, PWM, and
                logo.
              </>
            ),
            img: "/images/tutorial/fig-2c_tf_filter.png",
          },
          {
            title: "Discover Motifs Enriched in Gene Lists",
            text: (
              <>
                Run <strong>Motif Discovery for Genes</strong> (STREME). Select
                your gene list, optionally set promoter window and motif width
                (defaults to 6-10&nbsp;bp), then click <em>Run STREME</em>. Keep
                selected motifs in PWM (MEME) format to paste directly into
                Motif Viewer later.
              </>
            ),
            img: "/images/tutorial/fig-2d_streme.png",
          },
        ],
      },
      {
        id: "motif-viewer-setup",
        title: "Using Motif Viewer to Create Motif Hit Plots",
        figureHint: "Fig 3A-3D",
        steps: [
          {
            title: "Set Up Genomic Input",
            text: (
              <>
                For regular mode: Upload a <strong>gene list</strong> or type in
                your gene symbols, or upload a BED file with chromosomal
                coordinates. For Compare mode: Upload{" "}
                <strong>two gene lists</strong> (compare mode). Choose promoter
                window size around the TSS (default ±500&nbsp;bp). Click{" "}
                <em>Process Genomic Input</em> to validate and fetch
                annotations/sequences. If you see a warning with Unmatched Genes
                it is because of mismatch with the FlyBase annotations. Please
                consider using the most updated version of the gene name from
                FlyBase.
              </>
            ),
            imgs: [
              "/images/tutorial/fig-3a_genomic_input.png",
              "/images/tutorial/fig-3a_compareMode.png",
            ],
          },
          {
            title: "Provide Motif Input",
            text: (
              <>
                Add motifs as <strong>IUPAC</strong> or matrices{" "}
                <strong>(PWM/PFM/PCM)</strong> by pasting or uploading a .txt.
                If you're pasting a matrix, make sure to click{" "}
                <em>Parse Pasted Matrix</em> once you paste each matrix. Give
                each motif a name and color, then click <em>Process Motifs</em>{" "}
                once all your motifs are entered.
              </>
            ),
            imgs: [
              "/images/tutorial/fig-3b_motif_input.png",
              "/images/tutorial/fig-3c_motif_input_example.png",
            ],
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
            img: "/images/tutorial/fig-3c_fimo.png",
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
            img: "/images/tutorial/fig-3d_atac_chip.png",
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
                motif. ALWAYS click on <strong>Generate Plot</strong> to apply
                the settings and update the plot.
              </>
            ),
            img: "/images/tutorial/fig-4_settings1.png",
          },
          {
            title: "View Motif Hits Separately",
            text: (
              <>
                At the top right corner of the plot, you can see a motif legend.
                Click on a motif in the legend to hide its hits, and double
                click on a motif to isolate it and view only that motif's hits.
              </>
            ),
            imgs: [
              "/images/tutorial/fig-4_settings2.png",
              "/images/tutorial/fig-4_settings3.png",
            ],
          },
          {
            title: "Per-Motif Stringency",
            text: (
              <>
                Under <strong>Motif p-value thresholds</strong>, fine-tune
                individual motif thresholds with sliders to tighten or relax
                matches. Drag the slider to the left to make only the most
                significant matches remain.
              </>
            ),
            img: "/images/tutorial/fig-4_settingsMotif.png",
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
                strand, p-value, and score. Use the built-in toolbar at the top
                left to zoom, reset, and download the figure.
              </>
            ),
            img: "/images/tutorial/fig-4_plot_hover.png",
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
            imgs: [
              "/images/tutorial/fig-s2_overlay1.png",
              "/images/tutorial/fig-s2_overlay2.png",
            ],
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
                sets. For any further help/questions about VIDEO or bugs to be
                fixed, please contact vajay1@jh.edu and we will get back to you!
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
          <div className="position-sticky" style={{ top: 84 }}>
            <h5 className="mb-3">Tutorial</h5>
            <ListGroup>
              {sections.map((s) => (
                <ListGroup.Item key={s.id}>
                  <a
                    href={`#${s.id}`}
                    className="text-decoration-none"
                    onClick={(e) => {
                      e.preventDefault();
                      const el = document.getElementById(s.id);
                      if (el) el.scrollIntoView({ behavior: "smooth" });
                    }}
                  >
                    {s.title}
                  </a>
                  {s.figureHint && (
                    <div className="small text-muted">
                      <em>{s.figureHint}</em>
                    </div>
                  )}
                </ListGroup.Item>
              ))}
            </ListGroup>
          </div>
        </Col>

        {/* Content */}
        <Col lg={9}>
          <header className="mb-4">
            <h2 className="mb-2">Tutorial: Walkthrough Guide</h2>
            <p className="text-muted mb-0">
              From gene lists to motif discovery, motif-hit plots, and optional
              ATAC/ChIP overlays.
            </p>
          </header>

          {/* Plain, scrollable HTML-like content */}
          {sections.map((section, sIdx) => (
            <section id={section.id} key={section.id} className="mb-5">
              <h3 className="mb-2">{section.title}</h3>
              {section.figureHint && (
                <div className="small text-muted mb-3">
                  <em>{section.figureHint}</em>
                </div>
              )}

              {section.steps.map((step, i) => (
                <div key={`${section.id}-${i}`} className="mb-4">
                  <div className="d-flex align-items-center gap-2 mb-2">
                    <Badge bg="dark">
                      {sIdx + 1}.{i + 1}
                    </Badge>
                    <strong>{step.title}</strong>
                  </div>

                  <div className="mb-2">{step.text}</div>

                  <div className="d-flex flex-wrap align-items-start gap-3">
                    {step.ctaHref && (
                      <div>
                        <Button variant="primary" href={step.ctaHref}>
                          {step.ctaLabel ?? "Open"}
                        </Button>
                      </div>
                    )}

                    {/* Support multiple images or a single image */}
                    {step.imgs && step.imgs.length > 0 ? (
                      <div className="d-flex flex-wrap justify-content-start gap-3 w-100">
                        {step.imgs.map((src, idx) => (
                          <figure key={idx} style={{ maxWidth: 640 }}>
                            <Image
                              src={src}
                              alt={`${step.title} ${idx + 1}`}
                              fluid
                              rounded
                              style={{ border: "1px solid #eee" }}
                            />
                          </figure>
                        ))}
                      </div>
                    ) : (
                      step.img && (
                        <figure style={{ maxWidth: 640 }}>
                          <Image
                            src={step.img}
                            alt={step.title}
                            fluid
                            rounded
                            style={{ border: "1px solid #eee" }}
                          />
                        </figure>
                      )
                    )}
                  </div>
                </div>
              ))}

              {/* divider between sections */}
              <hr className="mt-4" />
            </section>
          ))}
        </Col>
      </Row>
    </Container>
  );
};

export default Tutorial;

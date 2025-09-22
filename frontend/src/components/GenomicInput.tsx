import React, {
  ChangeEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import InfoTip from "../components/InfoTip";
export type DataType = "bed" | "genes";
import {
  Spinner,
  Form,
  ButtonGroup,
  ToggleButton,
  Badge,
  Alert,
  Button,
} from "react-bootstrap";

export interface GenomicInputProps {
  dataType: DataType;
  setDataType: React.Dispatch<React.SetStateAction<DataType>>;

  bedFile: File | null;
  setBedFile: React.Dispatch<React.SetStateAction<File | null>>;

  geneListFile: File | null;
  setGeneListFile: React.Dispatch<React.SetStateAction<File | null>>;

  inputWindow: number;
  setInputWindow: React.Dispatch<React.SetStateAction<number>>;
  compareMode: boolean;
  canProcess?: boolean;

  // You likely call fetch('/get-genomic-input', { method: 'POST', body: FormData })
  // inside onProcess(). Keep that as-is; we’re just ensuring geneListFile exists.
  onProcess: () => Promise<void>;
}

type GeneInputMode = "upload" | "typed";

function parseGenesCaseSensitive(raw: string): string[] {
  // Split by commas/semicolons/whitespace/newlines; trim; drop empties; KEEP CASE
  // Optionally dedupe while preserving the FIRST occurrence (case-sensitive)
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of raw.split(/[\s,;]+/g)) {
    const g = token.trim();
    if (!g) continue;
    if (!seen.has(g)) {
      seen.add(g);
      out.push(g);
    }
  }
  return out;
}

export default function GenomicInput({
  dataType,
  setDataType,
  bedFile,
  setBedFile,
  geneListFile,
  setGeneListFile,
  inputWindow,
  setInputWindow,
  onProcess,
  compareMode,
  canProcess = true,
}: GenomicInputProps) {
  const [processed, setProcessed] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  // NEW: typed genes mode (case-sensitive)
  const [geneInputMode, setGeneInputMode] = useState<GeneInputMode>("upload");
  const [geneText, setGeneText] = useState<string>("");

  const typedGenes = useMemo(
    () => parseGenesCaseSensitive(geneText),
    [geneText]
  );

  // NEW: after processing, show any unmatched from backend
  const [unmatched, setUnmatched] = useState<string[]>([]);
  const [showUnmatched, setShowUnmatched] = useState<boolean>(true);

  // Allow parent to tell us the last backend payload if you wire it that way,
  // but for now we’ll intercept the Promise returned by onProcess() via a ref pattern.
  // If you already return the payload from onProcess(), you can adapt below.
  const lastProcessResolver = useRef<null | ((value: any) => void)>(null);

  // synthesize CSV file whenever typed genes change and “typed” mode is active
  useEffect(() => {
    if (dataType !== "genes" || geneInputMode !== "typed") return;

    if (typedGenes.length === 0) {
      setGeneListFile(null);
      return;
    }
    const csv = "gene\n" + typedGenes.join("\n"); // keep case exactly
    try {
      const f = new File([csv], "typed_genes.csv", { type: "text/csv" });
      setGeneListFile(f);
    } catch {
      const blob = new Blob([csv], { type: "text/csv" });
      // @ts-expect-error assign a name for downstream hints
      blob.name = "typed_genes.csv";
      setGeneListFile(blob as unknown as File);
    }
  }, [typedGenes, dataType, geneInputMode, setGeneListFile]);

  const handleProcess = async () => {
    setProcessed(false);
    setIsProcessing(true);
    setUnmatched([]);
    setShowUnmatched(true);

    try {
      // If your onProcess() returns the backend JSON, capture it; if not,
      // you can modify onProcess to return the parsed JSON. We fall back gracefully.
      const payloadOrVoid = await new Promise<any>(async (resolve) => {
        lastProcessResolver.current = resolve;
        const maybePayload = await onProcess();
        resolve(maybePayload);
      });

      // Try to read unmatched list if your onProcess returns the payload
      if (payloadOrVoid && Array.isArray(payloadOrVoid.unmatched_genes)) {
        setUnmatched(payloadOrVoid.unmatched_genes);
      }

      setProcessed(true);
    } catch (err: any) {
      // If backend returns 400 with JSON { unmatched_genes: [...], ... }, you may prefer
      // to plumb that through onProcess() so we can still show the list here.
      console.error(err);
    } finally {
      setIsProcessing(false);
    }
  };

  // Utility: download unmatched as CSV
  const downloadUnmatched = () => {
    const csv = "unmatched_gene\n" + unmatched.join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "unmatched_genes.csv";
    a.click();
    URL.revokeObjectURL(url);
  };
  const showFileChooser =
    dataType === "bed" || (dataType === "genes" && geneInputMode === "upload");

  if (compareMode) {
    return (
      <section className="mb-5 text-center">
        <div className="w-25 mx-auto mb-3">
          <label className="form-label d-block">
            Window size (bp)
            <InfoTip
              text="Distance ± from TSS for the genes in both lists."
              placement="right"
              id="genomic-input-info"
            />
          </label>
          <input
            type="number"
            className="form-control"
            value={inputWindow}
            min={50}
            step={50}
            onChange={(e) => {
              setInputWindow(+e.target.value);
              setProcessed(false);
            }}
          />
        </div>

        <div className="text-center">
          <button
            className="btn btn-success"
            onClick={handleProcess}
            disabled={isProcessing || !canProcess}
          >
            {isProcessing ? (
              <>
                <Spinner
                  as="span"
                  animation="border"
                  size="sm"
                  role="status"
                  aria-hidden="true"
                  className="me-2"
                />
                Processing…
              </>
            ) : (
              <>▶️ Process Genomic Input</>
            )}
          </button>
          {processed && !isProcessing && (
            <span className="badge bg-success ms-2">Processed!</span>
          )}
          {!canProcess && (
            <div className="text-muted small mt-2">
              Please provide both labels and CSVs above.
            </div>
          )}
        </div>
      </section>
    );
  }

  // --- Regular (single-list) UI ---
  return (
    <section className="mb-5 text-center">
      <div className="d-flex justify-content-center align-items-center gap-4 mb-4">
        {["bed", "genes"].map((type) => (
          <label
            key={type}
            className="form-check"
            style={{ cursor: "pointer" }}
          >
            <input
              type="radio"
              className="form-check-input me-2"
              checked={dataType === type}
              onChange={() => {
                setDataType(type as DataType);
                setProcessed(false);
                setUnmatched([]);
                setShowUnmatched(true);
                if (type === "bed") {
                  setGeneText("");
                  setGeneListFile(null);
                  setGeneInputMode("upload");
                } else {
                  setBedFile(null);
                }
              }}
            />
            <span className="form-check-label">
              {type === "bed" ? "BED file" : "Gene list"}
            </span>
          </label>
        ))}
      </div>

      {/* Upload vs Type genes toggle */}
      {dataType === "genes" && (
        <div className="mb-3">
          <ButtonGroup aria-label="Gene input mode">
            <ToggleButton
              id="gene-upload"
              type="radio"
              variant="outline-primary"
              name="gene-input-mode"
              value="upload"
              checked={geneInputMode === "upload"}
              onChange={() => {
                setGeneInputMode("upload");
                setProcessed(false);
              }}
            >
              Upload CSV
            </ToggleButton>
            <ToggleButton
              id="gene-typed"
              type="radio"
              variant="outline-primary"
              name="gene-input-mode"
              value="typed"
              checked={geneInputMode === "typed"}
              onChange={() => {
                setGeneInputMode("typed");
                setProcessed(false);
                setGeneListFile(null); // will be synthesized from text
              }}
            >
              Type genes
            </ToggleButton>
          </ButtonGroup>
          <div className="small text-muted mt-2">
            {geneInputMode === "typed"
              ? "Paste or type symbols separated by spaces, commas, or new lines. Case is preserved."
              : "Upload a CSV with one gene symbol per row."}
          </div>
        </div>
      )}

      <div className="mx-auto" style={{ maxWidth: 520 }}>
        {showFileChooser && (
          <div className="mx-auto" style={{ maxWidth: 520 }}>
            <Form.Group className="mb-4 text-center">
              <Form.Control
                type="file"
                accept={dataType === "bed" ? ".bed" : ".csv"}
                onChange={(e: ChangeEvent<HTMLInputElement>) => {
                  const f = e.currentTarget.files?.[0] ?? null;
                  if (dataType === "bed") setBedFile(f);
                  else setGeneListFile(f);
                  setProcessed(false);
                  setUnmatched([]);
                  setShowUnmatched(true);
                }}
                className="d-block mx-auto"
              />
              <Form.Text muted>
                {dataType === "bed"
                  ? bedFile?.name || "Choose a .bed file"
                  : geneListFile?.name || "Choose a .csv gene list"}
              </Form.Text>
            </Form.Group>
          </div>
        )}

        {/* Typed genes textarea */}
        {dataType === "genes" && geneInputMode === "typed" && (
          <>
            <Form.Group className="mb-3 text-start">
              <Form.Label>
                Gene symbols
                <InfoTip
                  id="typed-genes-tip"
                  placement="right"
                  text="Separate by spaces, commas, or new lines. Case-sensitive; duplicates removed keeping first occurrence."
                />
              </Form.Label>
              <Form.Control
                as="textarea"
                rows={4}
                value={geneText}
                placeholder="e.g., CrebA, otp, byn, Retn"
                onChange={(e) => {
                  setGeneText(e.target.value);
                  setProcessed(false);
                  setUnmatched([]);
                  setShowUnmatched(true);
                }}
              />
            </Form.Group>

            {/* Preview */}
            <div className="text-start mb-4">
              <div className="small text-muted mb-1">
                {typedGenes.length === 0
                  ? "No genes detected yet."
                  : `${typedGenes.length} genes detected:`}
              </div>
              {typedGenes.length > 0 && (
                <div className="d-flex flex-wrap gap-2">
                  {typedGenes.slice(0, 12).map((g) => (
                    <Badge bg="secondary" key={g}>
                      {g}
                    </Badge>
                  ))}
                  {typedGenes.length > 12 && (
                    <span className="small text-muted">
                      +{typedGenes.length - 12} more
                    </span>
                  )}
                </div>
              )}
            </div>
          </>
        )}
      </div>

      <div className="w-25 mx-auto mb-3">
        <label className="form-label d-block">
          Window size (bp)
          <InfoTip
            text="± distance from TSS (for gene list) or from peak midpoint (for BED)."
            placement="right"
            id="genomic-input-info"
          />
        </label>
        <input
          type="number"
          className="form-control"
          value={inputWindow}
          min={50}
          step={50}
          onChange={(e) => {
            setInputWindow(+e.target.value);
            setProcessed(false);
            setUnmatched([]);
            setShowUnmatched(true);
          }}
        />
      </div>

      {/* Unmatched banner */}
      {unmatched.length > 0 && showUnmatched && (
        <div className="mx-auto" style={{ maxWidth: 720 }}>
          <Alert
            variant="warning"
            onClose={() => setShowUnmatched(false)}
            dismissible
          >
            <Alert.Heading>
              Some genes didn't match the annotation
            </Alert.Heading>
            <div className="mb-2">
              {`Gene symbols are case-sensitive - please check your entries! Found ${
                unmatched.length
              } unmatched gene${unmatched.length > 1 ? "s" : ""}.`}
            </div>
            <div className="small">
              {unmatched.slice(0, 25).map((g) => (
                <Badge bg="warning" text="dark" className="me-2 mb-2" key={g}>
                  {g}
                </Badge>
              ))}
              {unmatched.length > 25 && (
                <span className="text-muted">
                  +{unmatched.length - 25} more
                </span>
              )}
            </div>
            <div className="mt-3">
              <Button
                size="sm"
                variant="outline-secondary"
                onClick={downloadUnmatched}
              >
                Download unmatched CSV
              </Button>
            </div>
          </Alert>
        </div>
      )}

      <div className="text-center">
        <button
          className="btn btn-success"
          onClick={handleProcess}
          disabled={
            isProcessing ||
            !canProcess ||
            (dataType === "genes" &&
              geneInputMode === "typed" &&
              typedGenes.length === 0 &&
              !geneListFile)
          }
        >
          {isProcessing ? (
            <>
              <Spinner
                as="span"
                animation="border"
                size="sm"
                role="status"
                aria-hidden="true"
                className="me-2"
              />
              Processing…
            </>
          ) : (
            <>▶️ Process Genomic Input</>
          )}
        </button>
        {processed && !isProcessing && (
          <span className="badge bg-success ms-2">Processed!</span>
        )}
      </div>
    </section>
  );
}

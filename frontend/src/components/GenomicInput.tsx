import React, { ChangeEvent, useState } from "react";
import InfoTip from "../components/InfoTip";
export type DataType = "bed" | "genes";
import { Spinner, Form } from "react-bootstrap";

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

  onProcess: () => Promise<void>;
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
  const handleProcess = async () => {
    setProcessed(false);
    setIsProcessing(true);
    try {
      await onProcess(); // wait for the real work
      setProcessed(true); // only now mark “done”
    } catch (err) {
      console.error(err);
      // you might want to show an error message here
    } finally {
      setIsProcessing(false);
    }
  };

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

  // --- Regular (single-list) UI: keep existing controls ---
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
              }}
            />
            <span className="form-check-label">
              {type === "bed" ? "BED file" : "Gene list (CSV)"}
            </span>
          </label>
        ))}
      </div>

      <div className="mx-auto" style={{ maxWidth: 300 }}>
        <Form.Group className="mb-4 text-center">
          <Form.Control
            type="file"
            accept={dataType === "bed" ? ".bed" : ".csv"}
            onChange={(e: ChangeEvent<HTMLInputElement>) => {
              const f = e.currentTarget.files?.[0] ?? null;
              if (dataType === "bed") setBedFile(f);
              else setGeneListFile(f);
              setProcessed(false);
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
      </div>
    </section>
  );
}

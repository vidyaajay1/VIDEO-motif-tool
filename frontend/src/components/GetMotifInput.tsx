import React, { CSSProperties, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Form,
  Button,
  InputGroup,
  Card,
  Row,
  Col,
  Spinner,
} from "react-bootstrap";
import { FaTrash } from "react-icons/fa";

export type UserMotif = {
  type: "iupac" | "pwm" | "pcm";
  iupac: string;
  pwm: number[][];
  pcm: number[][];
  name: string;
  color: string;
};

interface GetMotifInputProps {
  motif: UserMotif;
  onChange: (patch: Partial<UserMotif>) => void;
  onRemove: () => void;
}

const cardStyle: CSSProperties = {
  maxWidth: "800px",
};

export default function GetMotifInput({
  motif,
  onChange,
  onRemove,
}: GetMotifInputProps) {
  const [matrixPaste, setMatrixPaste] = useState("");
  const [showMatrixEditor, setShowMatrixEditor] = useState(true);

  const handleTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = e.target.value as UserMotif["type"];
    onChange({
      type: newType,
      pwm: [],
      pcm: [],
      iupac: "",
    });
    setShowMatrixEditor(true);
  };

  const handleAddRow = () => {
    const rows = motif.type === "pwm" ? motif.pwm : motif.pcm;
    const newRow =
      motif.type === "pwm" ? [0.25, 0.25, 0.25, 0.25] : [1, 1, 1, 1];
    onChange({ [motif.type]: [...rows, newRow] } as any);
  };

  const handleRemoveRow = (idx: number) => {
    const rows = motif.type === "pwm" ? motif.pwm : motif.pcm;
    onChange({
      [motif.type]: rows.filter((_, i) => i !== idx),
    } as any);
  };

  const handleCellChange = (rowIdx: number, colIdx: number, value: number) => {
    const rows = motif.type === "pwm" ? motif.pwm : motif.pcm;
    const updated = rows.map((r, i) =>
      i === rowIdx ? r.map((v, j) => (j === colIdx ? value : v)) : r
    );
    onChange({ [motif.type]: updated } as any);
  };

  const parseMatrixText = (text: string) => {
    const lines = text
      .trim()
      .split(/\r?\n/)
      // skip any line that doesn’t begin with a digit, minus, or dot
      .filter((line) => /^\s*[\d\.\-]/.test(line));

    const matrix = lines.map((line) =>
      line
        .trim()
        .split(/\s+/)
        .map((token) => parseFloat(token))
    );

    onChange({ pwm: [], pcm: [], [motif.type]: matrix } as any);
  };

  const handlePasteSubmit = () => {
    parseMatrixText(matrixPaste);
    setShowMatrixEditor(false);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      parseMatrixText(text);
      setShowMatrixEditor(false);
    };
    reader.readAsText(file);
  };

  return (
    <section>
      <Card border="secondary" className="mb-4 mx-auto" style={cardStyle}>
        <Card.Body>
          <Row className="align-items-center mb-3">
            <Col xs="auto">
              <Form.Select
                value={motif.type}
                onChange={handleTypeChange}
                size="sm"
                style={{ backgroundColor: "#e9f7ef", borderColor: "#4caf50" }}
              >
                <option value="iupac">IUPAC</option>
                <option value="pwm">PWM/PFM</option>
                <option value="pcm">PCM</option>
              </Form.Select>
            </Col>

            {(motif.type === "pwm" || motif.type === "pcm") &&
              !showMatrixEditor && (
                <>
                  <Col>
                    <strong>Motif Parsed:</strong> {motif.name || "Unnamed"}
                  </Col>
                  <Col xs="auto">
                    <Button
                      variant="outline-secondary"
                      size="sm"
                      onClick={() => setShowMatrixEditor(true)}
                    >
                      Edit Matrix
                    </Button>
                  </Col>
                </>
              )}

            {motif.type === "iupac" && (
              <Col>
                <Form.Control
                  placeholder="IUPAC sequence"
                  size="sm"
                  value={motif.iupac}
                  onChange={(e) => onChange({ iupac: e.target.value })}
                />
              </Col>
            )}
          </Row>

          {(motif.type === "pwm" || motif.type === "pcm") &&
            showMatrixEditor && (
              <div className="mb-3">
                <Button
                  variant="outline-secondary"
                  size="sm"
                  className="mb-2"
                  onClick={handleAddRow}
                >
                  + Add row
                </Button>

                {(motif[motif.type] as number[][]).map((row, rIdx) => (
                  <InputGroup className="mb-2" key={rIdx}>
                    {row.map((cell, cIdx) => (
                      <Form.Control
                        key={cIdx}
                        type="number"
                        step={motif.type === "pwm" ? 0.01 : 1}
                        size="sm"
                        value={cell}
                        onChange={(e) =>
                          handleCellChange(
                            rIdx,
                            cIdx,
                            parseFloat(e.target.value) || 0
                          )
                        }
                      />
                    ))}
                    <Button
                      variant="outline-danger"
                      size="sm"
                      onClick={() => handleRemoveRow(rIdx)}
                    >
                      –
                    </Button>
                  </InputGroup>
                ))}

                <Form.Control
                  as="textarea"
                  rows={5}
                  placeholder="Paste motif matrix here (vertical ACGT), no header"
                  value={matrixPaste}
                  onChange={(e) => setMatrixPaste(e.target.value)}
                  className="mb-2"
                />
                <Button
                  size="sm"
                  variant="outline-primary"
                  onClick={handlePasteSubmit}
                  className="me-2"
                >
                  Parse Pasted Matrix
                </Button>

                <Form.Group controlId="formFile" className="mt-2">
                  <Form.Label>Or upload motif file</Form.Label>
                  <Form.Control
                    type="file"
                    accept=".txt"
                    onChange={handleFileUpload}
                    size="sm"
                  />
                </Form.Group>

                <Button
                  size="sm"
                  variant="outline-secondary"
                  className="mt-2"
                  onClick={() => setShowMatrixEditor(false)}
                >
                  Collapse Editor
                </Button>
              </div>
            )}

          <Row className="align-items-center">
            <Col>
              <Form.Control
                placeholder="Motif name"
                size="sm"
                value={motif.name}
                onChange={(e) => onChange({ name: e.target.value })}
              />
            </Col>
            <Col xs="auto">
              <InputGroup>
                <InputGroup.Text style={{ padding: "0.25rem 0.5rem" }}>
                  <i className="bi bi-palette-fill" aria-hidden="true" />
                </InputGroup.Text>
                <Form.Control
                  type="color"
                  title="Choose color"
                  size="sm"
                  value={motif.color}
                  onChange={(e) => onChange({ color: e.target.value })}
                  style={{
                    padding: 0,
                    border: "none",
                    width: "2rem",
                    height: "2rem",
                  }}
                />
              </InputGroup>
            </Col>
            <Col xs="auto">
              <Button variant="light" size="sm" onClick={onRemove}>
                <FaTrash />
              </Button>
            </Col>
          </Row>
        </Card.Body>
      </Card>
    </section>
  );
}

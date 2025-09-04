// components/MotifOccurencePlot.tsx (trimmed)
import Plot from "react-plotly.js";
import FiltersBar from "./FiltersBar";
import { useMotifViewer } from "../context/MotifViewerContext";
import React from "react";
import type { FilterSettings } from "../types/FilterSettings";

export default function MotifOccurencePlot({
  figureJson,
  onApplyFilters,
  motifList = [],
}: {
  figureJson?: string;
  onApplyFilters: (filters: FilterSettings) => void | Promise<void>;
  motifList?: string[];
}) {
  const { fimoThreshold } = useMotifViewer();

  const parsed = React.useMemo(() => {
    if (!figureJson) return null;
    try {
      return JSON.parse(figureJson);
    } catch {
      return null;
    }
  }, [figureJson]);

  return (
    <section className="mb-5">
      <h4 className="text-center mb-3 mt-3">Motif Occurence Overview</h4>

      <FiltersBar
        motifList={motifList}
        fimoThreshold={fimoThreshold}
        onApply={onApplyFilters}
      />

      <div className="d-flex justify-content-center">
        {parsed ? (
          <Plot
            data={parsed.data ?? []}
            layout={{ ...parsed.layout, autosize: true }}
            config={{ responsive: true, scrollZoom: true, displaylogo: false }}
            style={{
              width: parsed.layout?.width ?? 1000,
              height: parsed.layout?.height ?? 900,
            }}
            useResizeHandler
          />
        ) : (
          <div>No plot generated yet.</div>
        )}
      </div>
    </section>
  );
}

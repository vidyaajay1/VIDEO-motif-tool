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
  children,
}: {
  figureJson?: string;
  onApplyFilters: (filters: FilterSettings) => void | Promise<void>;
  motifList?: string[];
  children?: React.ReactNode;
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
      {/*  Download button slot: right below Generate Plot */}
      {parsed && children ? (
        <div className="d-flex gap-2 mt-2">{children}</div>
      ) : null}
      {parsed && (
        <div className="text-muted mb-1" style={{ fontSize: "0.8rem" }}>
          💡 Click any motif bar to view the locus in FlyBase JBrowse
        </div>
      )}
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
            onClick={(eventData: any) => {
              const point = eventData?.points?.[0];
              if (!point) return;
              const customdata = point.customdata;
              if (!customdata) return;
              // JBrowse URL is at index 10 (or 9 if no sequence column)
              // Try index 10 first, fall back to checking if it looks like a URL
              const url = Array.isArray(customdata)
                ? customdata.find(
                    (v: any) =>
                      typeof v === "string" &&
                      v.startsWith("https://flybase.org"),
                  )
                : null;
              if (url) window.open(url, "_blank", "noopener,noreferrer");
            }}
          />
        ) : (
          <div>No plot generated yet. Click on Generate Plot!</div>
        )}
      </div>
    </section>
  );
}

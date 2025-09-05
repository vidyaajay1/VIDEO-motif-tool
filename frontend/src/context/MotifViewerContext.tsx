// MotifViewerContext.tsx
import React, { createContext, useContext, useState, ReactNode } from "react";
import { UserMotif } from "../components/GetMotifInput";

type DataType = "bed" | "genes";
type CompareResults = {
  [sessionId: string]: {
    filtered: boolean;
    processedIds: string[]; // data_ids returned by /filter-motif-hits-batch
  };
};
type DataIdLabels = Record<string, string>;

interface MotifViewerContextType {
  dataType: DataType;
  setDataType: React.Dispatch<React.SetStateAction<DataType>>;
  bedFile: File | null;
  setBedFile: React.Dispatch<React.SetStateAction<File | null>>;
  geneListFile: File | null;
  setGeneListFile: React.Dispatch<React.SetStateAction<File | null>>;
  inputWindow: number;
  setInputWindow: React.Dispatch<React.SetStateAction<number>>;
  dataId: string | null;
  setDataId: React.Dispatch<React.SetStateAction<string | null>>;
  peakList: string[];
  setPeakList: React.Dispatch<React.SetStateAction<string[]>>;

  motifs: UserMotif[];
  setMotifs: React.Dispatch<React.SetStateAction<UserMotif[]>>;

  scanComplete: boolean;
  setScanComplete: React.Dispatch<React.SetStateAction<boolean>>;
  overviewUrl: string | null;
  setOverviewUrl: React.Dispatch<React.SetStateAction<string | null>>;

  filteredOverviewUrl: string | null;
  setFilteredOverviewUrl: React.Dispatch<React.SetStateAction<string | null>>;

  /** NEW: FIMO p-value threshold shared across steps (as string to match inputs) */
  fimoThreshold: string;
  setFimoThreshold: React.Dispatch<React.SetStateAction<string>>;

  compareResults: CompareResults;
  setCompareResults: React.Dispatch<React.SetStateAction<CompareResults>>;
  setCompareSessionResults: (
    sessionId: string,
    data: { filtered: boolean; processedIds: string[] }
  ) => void;

  singleTopHitsReady: boolean;
  setSingleTopHitsReady: React.Dispatch<React.SetStateAction<boolean>>;
  labelsByDataId: DataIdLabels;
  setLabelsByDataId: React.Dispatch<React.SetStateAction<DataIdLabels>>;

  fetchJSON: (
    url: string,
    opts: RequestInit,
    onError: (msg: string) => void
  ) => Promise<any>;
}

const MotifViewerContext = createContext<MotifViewerContextType | undefined>(
  undefined
);

export const useMotifViewer = () => {
  const ctx = useContext(MotifViewerContext);
  if (!ctx)
    throw new Error("useMotifViewer must be used within a MotifViewerProvider");
  return ctx;
};

export const MotifViewerProvider = ({ children }: { children: ReactNode }) => {
  const [dataType, setDataType] = useState<DataType>("bed");
  const [bedFile, setBedFile] = useState<File | null>(null);
  const [geneListFile, setGeneListFile] = useState<File | null>(null);
  const [inputWindow, setInputWindow] = useState<number>(500);
  const [dataId, setDataId] = useState<string | null>(null);
  const [peakList, setPeakList] = useState<string[]>([]);

  const [motifs, setMotifs] = useState<UserMotif[]>([
    {
      type: "iupac",
      iupac: "",
      pwm: [[0.25, 0.25, 0.25, 0.25]],
      pcm: [[1, 1, 1, 1]],
      name: "",
      color: "#000000",
    },
  ]);

  const [scanComplete, setScanComplete] = useState(false);
  const [overviewUrl, setOverviewUrl] = useState<string | null>(null);
  const [filteredOverviewUrl, setFilteredOverviewUrl] = useState<string | null>(
    null
  );

  /** NEW: shared FIMO threshold (default 1e-3 to match current UI) */
  const [fimoThreshold, setFimoThreshold] = useState<string>("0.001");
  const [labelsByDataId, setLabelsByDataId] = useState<DataIdLabels>({});
  const fetchJSON = async (
    url: string,
    opts: RequestInit,
    onError: (msg: string) => void
  ) => {
    try {
      const res = await fetch(url, opts);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    } catch (e: any) {
      onError(e.message);
      return null;
    }
  };

  const [compareResults, setCompareResults] = useState<CompareResults>({});
  const [singleTopHitsReady, setSingleTopHitsReady] = useState(false);

  const setCompareSessionResults = (
    sessionId: string,
    data: { filtered: boolean; processedIds: string[] }
  ) => {
    setCompareResults((prev) => ({ ...prev, [sessionId]: data }));
  };

  return (
    <MotifViewerContext.Provider
      value={{
        dataType,
        setDataType,
        bedFile,
        setBedFile,
        geneListFile,
        setGeneListFile,
        inputWindow,
        setInputWindow,
        dataId,
        setDataId,
        peakList,
        setPeakList,
        motifs,
        setMotifs,
        scanComplete,
        setScanComplete,
        overviewUrl,
        setOverviewUrl,
        filteredOverviewUrl,
        setFilteredOverviewUrl,
        fimoThreshold, // NEW
        setFimoThreshold, // NEW
        compareResults,
        setCompareResults,
        setCompareSessionResults,
        singleTopHitsReady,
        setSingleTopHitsReady,
        labelsByDataId, // ✅ ADDED
        setLabelsByDataId, // ✅ ADDED
        fetchJSON,
      }}
    >
      {children}
    </MotifViewerContext.Provider>
  );
};

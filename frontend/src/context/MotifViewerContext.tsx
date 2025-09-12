import React, { createContext, useContext, useState, ReactNode } from "react";
import { UserMotif } from "../components/GetMotifInput";

type DataType = "bed" | "genes";
type CompareResults = {
  [sessionId: string]: {
    filtered: boolean;
    processedIds: string[];
  };
};
type DataIdLabels = Record<string, string>;

interface MotifViewerContextType {
  dataType: DataType;
  setDataType: React.Dispatch<React.SetStateAction<DataType>>;

  bedFile: File | null;
  setBedFile: React.Dispatch<React.SetStateAction<File | null>>;

  // (legacy single-list fields; keep for back-compat)
  geneListFile: File | null;
  setGeneListFile: React.Dispatch<React.SetStateAction<File | null>>;
  dataId: string | null;
  setDataId: React.Dispatch<React.SetStateAction<string | null>>;

  // compare mode state
  isCompare: boolean;
  setIsCompare: React.Dispatch<React.SetStateAction<boolean>>;
  labelA: string;
  setLabelA: React.Dispatch<React.SetStateAction<string>>;
  labelB: string;
  setLabelB: React.Dispatch<React.SetStateAction<string>>;
  geneListFileA: File | null;
  setGeneListFileA: React.Dispatch<React.SetStateAction<File | null>>;
  geneListFileB: File | null;
  setGeneListFileB: React.Dispatch<React.SetStateAction<File | null>>;
  dataIdA: string | null;
  setDataIdA: React.Dispatch<React.SetStateAction<string | null>>;
  dataIdB: string | null;
  setDataIdB: React.Dispatch<React.SetStateAction<string | null>>;

  inputWindow: number;
  setInputWindow: React.Dispatch<React.SetStateAction<number>>;
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

  fimoThreshold: string;
  setFimoThreshold: React.Dispatch<React.SetStateAction<string>>;

  labelsByDataId: DataIdLabels;
  setLabelsByDataId: React.Dispatch<React.SetStateAction<DataIdLabels>>;

  compareResults: CompareResults;
  setCompareResults: React.Dispatch<React.SetStateAction<CompareResults>>;
  setCompareSessionResults: (
    sessionId: string,
    data: { filtered: boolean; processedIds: string[] }
  ) => void;

  singleTopHitsReady: boolean;
  setSingleTopHitsReady: React.Dispatch<React.SetStateAction<boolean>>;
  sessionId: string | null;
  setSessionId: React.Dispatch<React.SetStateAction<string | null>>;

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

  // legacy single-list
  const [geneListFile, setGeneListFile] = useState<File | null>(null);
  const [dataId, setDataId] = useState<string | null>(null);

  // compare mode
  const [isCompare, setIsCompare] = useState(false);
  const [labelA, setLabelA] = useState("List A");
  const [labelB, setLabelB] = useState("List B");
  const [geneListFileA, setGeneListFileA] = useState<File | null>(null);
  const [geneListFileB, setGeneListFileB] = useState<File | null>(null);
  const [dataIdA, setDataIdA] = useState<string | null>(null);
  const [dataIdB, setDataIdB] = useState<string | null>(null);

  const [inputWindow, setInputWindow] = useState<number>(500);
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

  const [fimoThreshold, setFimoThreshold] = useState<string>("0.001");
  const [labelsByDataId, setLabelsByDataId] = useState<DataIdLabels>({});

  const [compareResults, setCompareResults] = useState<CompareResults>({});
  const [singleTopHitsReady, setSingleTopHitsReady] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
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

  const setCompareSessionResults = (
    sessionId: string,
    data: { filtered: boolean; processedIds: string[] }
  ) => setCompareResults((prev) => ({ ...prev, [sessionId]: data }));

  // light persistence across refresh
  React.useEffect(() => {
    const saved = sessionStorage.getItem("mv.compare");
    if (saved) {
      const s = JSON.parse(saved);
      setIsCompare(!!s.isCompare);
      setLabelA(s.labelA ?? "List A");
      setLabelB(s.labelB ?? "List B");
      setDataIdA(s.dataIdA ?? null);
      setDataIdB(s.dataIdB ?? null);
    }
  }, []);
  React.useEffect(() => {
    console.log("[MotifViewerProvider] mounted");
    return () => console.log("[MotifViewerProvider] unmounted");
  }, []);
  React.useEffect(() => {
    sessionStorage.setItem(
      "mv.compare",
      JSON.stringify({ isCompare, labelA, labelB, dataIdA, dataIdB })
    );
  }, [isCompare, labelA, labelB, dataIdA, dataIdB]);

  return (
    <MotifViewerContext.Provider
      value={{
        dataType,
        setDataType,
        bedFile,
        setBedFile,

        geneListFile,
        setGeneListFile, // legacy
        dataId,
        setDataId, // legacy

        isCompare,
        setIsCompare,
        labelA,
        setLabelA,
        labelB,
        setLabelB,
        geneListFileA,
        setGeneListFileA,
        geneListFileB,
        setGeneListFileB,
        dataIdA,
        setDataIdA,
        dataIdB,
        setDataIdB,

        inputWindow,
        setInputWindow,
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

        fimoThreshold,
        setFimoThreshold,

        labelsByDataId,
        setLabelsByDataId,

        compareResults,
        setCompareResults,
        setCompareSessionResults,
        singleTopHitsReady,
        setSingleTopHitsReady,
        sessionId,
        setSessionId,
        fetchJSON,
      }}
    >
      {children}
    </MotifViewerContext.Provider>
  );
};

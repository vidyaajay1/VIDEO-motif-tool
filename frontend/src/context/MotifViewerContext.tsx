// MotifViewerContext.tsx
import React, { createContext, useContext, useState, ReactNode } from "react";
import { UserMotif } from "../components/GetMotifInput";
import { FilterSettings } from "../components/MotifOccurencePlot";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

// -------------------- Types --------------------
type DataType = "bed" | "genes";

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

  fetchJSON: (
    url: string,
    opts: RequestInit,
    onError: (msg: string) => void
  ) => Promise<any>;
}

// -------------------- Context --------------------
const MotifViewerContext = createContext<MotifViewerContextType | undefined>(
  undefined
);

export const useMotifViewer = () => {
  const context = useContext(MotifViewerContext);
  if (!context) {
    throw new Error("useMotifViewer must be used within a MotifViewerProvider");
  }
  return context;
};

// -------------------- Provider --------------------
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
        fetchJSON,
      }}
    >
      {children}
    </MotifViewerContext.Provider>
  );
};

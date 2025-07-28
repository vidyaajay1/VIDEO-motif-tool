import React, { createContext, useContext, useReducer } from "react";

type State = {
  dataSource: string;
  stage: string;
  tissue: string;
  tissueOptions: string[];
  genes: { gene: string; avg_log2FC: number }[];
  filteredTFs: {
    symbol: string;
    flybase_id: string;
    motif_id: string | null;
    pfm: number[][] | null;
  }[];
};

type Action =
  | { type: "SET_DATASOURCE"; payload: string }
  | { type: "SET_STAGE"; payload: string }
  | { type: "SET_TISSUE"; payload: string }
  | { type: "SET_TISSUE_OPTIONS"; payload: string[] }
  | { type: "SET_GENES"; payload: { gene: string; avg_log2FC: number }[] }
  | {
      type: "SET_FILTERED_TFS";
      payload: {
        symbol: string;
        flybase_id: string;
        motif_id: string | null;
        pfm: number[][] | null;
      }[];
    };

const initialState: State = {
  dataSource: "peng2024",
  stage: "",
  tissue: "",
  tissueOptions: [],
  genes: [],
  filteredTFs: [],
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "SET_DATASOURCE":
      return { ...state, dataSource: action.payload };
    case "SET_STAGE":
      return {
        ...state,
        stage: action.payload,
        tissue: "",
        tissueOptions: [],
        genes: [],
      };
    case "SET_TISSUE":
      return { ...state, tissue: action.payload };
    case "SET_TISSUE_OPTIONS":
      return { ...state, tissueOptions: action.payload };
    case "SET_GENES":
      return { ...state, genes: action.payload };
    case "SET_FILTERED_TFS":
      return { ...state, filteredTFs: action.payload };

    default:
      return state;
  }
}

const TFContext = createContext<{
  state: State;
  dispatch: React.Dispatch<Action>;
}>({ state: initialState, dispatch: () => null });

export const TFProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <TFContext.Provider value={{ state, dispatch }}>
      {children}
    </TFContext.Provider>
  );
};

export const useTFContext = () => useContext(TFContext);

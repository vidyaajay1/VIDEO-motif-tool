import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { TFProvider } from "./context/TFContext";
import Layout from "./components/Layout";
import MotifViewer from "./pages/MotifViewer";
import TFFinder from "./pages/TFFinder";
import { MotifViewerProvider } from "./context/MotifViewerContext";

function App() {
  return (
    <Router>
      <TFProvider>
        <MotifViewerProvider>
          {" "}
          {/* ✅ Wrap it here */}
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<MotifViewer />} />
              <Route path="tf-finder" element={<TFFinder />} />
            </Route>
          </Routes>
        </MotifViewerProvider>
      </TFProvider>
    </Router>
  );
}
export default App;

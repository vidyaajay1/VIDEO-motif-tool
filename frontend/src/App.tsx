import React, { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Outlet,
} from "react-router-dom";
import { TFProvider } from "./context/TFContext";
import { MotifViewerProvider } from "./context/MotifViewerContext";

import CollapsibleSidebar from "./components/CollapsibleSidebar";
import MotifViewer from "./pages/MotifViewer";
import TFFinder from "./pages/TFFinder";
import Tutorial from "./pages/Tutorial"; // ✅ new page

// ✅ Shell that replaces Layout: sidebar + main content via <Outlet/>
function AppShell() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="d-flex" style={{ minHeight: "100vh" }}>
      <div
        style={{
          width: collapsed ? 64 : 240,
          transition: "width 0.2s ease",
        }}
      >
        <CollapsibleSidebar collapsed={collapsed} onToggle={setCollapsed} />
      </div>

      {/* ✅ Make this scrollable */}
      <div
        className="flex-grow-1 bg-light"
        style={{ height: "100vh", overflowY: "auto" }}
      >
        <div className="container-fluid py-3">
          <Outlet />
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <TFProvider>
        <MotifViewerProvider>
          <Routes>
            <Route path="/" element={<AppShell />}>
              <Route index element={<MotifViewer />} />
              <Route path="tf-finder" element={<TFFinder />} />
              <Route path="tutorial" element={<Tutorial />} />{" "}
              {/* ✅ new route */}
            </Route>
          </Routes>
        </MotifViewerProvider>
      </TFProvider>
    </Router>
  );
}

export default App;

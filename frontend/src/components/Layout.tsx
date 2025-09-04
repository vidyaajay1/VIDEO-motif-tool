import React from "react";
import CollapsibleSidebar from "./CollapsibleSidebar";
import { Outlet } from "react-router-dom";
import "../types/Layout.css";

const Layout: React.FC = () => {
  const [collapsed, setCollapsed] = React.useState(false);

  const handleToggle = (next: boolean) => {
    setCollapsed(next);
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  };
  const sidebarWidth = collapsed ? 80 : 200; // px

  return (
    <div className={`app-wrapper ${collapsed ? "is-collapsed" : ""}`}>
      <div
        className="sidebar"
        style={{
          width: sidebarWidth,
          flex: `0 0 ${sidebarWidth}px`,
          transition: "width .3s ease, flex-basis .3s ease",
          boxSizing: "border-box",
        }}
      >
        <CollapsibleSidebar collapsed={collapsed} onToggle={handleToggle} />
      </div>

      <div className="main-content">
        <Outlet />
      </div>
    </div>
  );
};

export default Layout;

import React from "react";
import CollapsibleSidebar from "./CollapsibleSidebar";
import { Outlet } from "react-router-dom";

const Layout: React.FC = () => {
  return (
    <div className="app-wrapper">
      <div className="sidebar">
        <CollapsibleSidebar />
      </div>
      <div className="main-content">
        <Outlet />
      </div>
    </div>
  );
};

export default Layout;

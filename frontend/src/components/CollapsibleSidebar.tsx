import React, { CSSProperties } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { FaBars, FaEye, FaSearch, FaBookOpen } from "react-icons/fa"; // ✅ added FaBookOpen
import { Nav } from "react-bootstrap";
import { Link } from "react-router-dom";

export default function CollapsibleSidebar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: (next: boolean) => void;
}) {
  const isOpen = !collapsed;

  const containerStyle: React.CSSProperties = {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    overflow: "hidden",
  };

  const linkStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    justifyContent: isOpen ? "flex-start" : "center",
    padding: "0.5rem 0",
    width: "100%",
  };

  return (
    <div className="bg-dark text-white" style={containerStyle}>
      <button
        className="btn btn-outline-light btn-sm mt-3"
        onClick={() => onToggle(!collapsed)}
        aria-label="Toggle Sidebar"
      >
        <FaBars />
      </button>

      <div className="d-flex justify-content-center align-items-center mt-3 mb-3 text-center px-2">
        <strong style={{ fontSize: isOpen ? "1rem" : "0.8rem" }}>
          {isOpen
            ? "Visual Integration of Drosophila Enhancer Organization"
            : "VIDEO"}
        </strong>
      </div>

      <Nav className="flex-column w-100">
        <Nav.Link
          as={Link}
          to="/"
          className="text-white"
          style={linkStyle}
          title="Motif Viewer"
        >
          <FaEye className="me-2 ms-3" />
          {isOpen && "Motif Viewer"}
        </Nav.Link>

        <Nav.Link
          as={Link}
          to="/tf-finder"
          className="text-white"
          style={linkStyle}
          title="TF Finder"
        >
          <FaSearch className="me-2 ms-3" />
          {isOpen && "TF Finder"}
        </Nav.Link>

        {/* ✅ New Tutorial link */}
        <Nav.Link
          as={Link}
          to="/tutorial"
          className="text-white"
          style={linkStyle}
          title="Tutorial"
        >
          <FaBookOpen className="me-2 ms-3" />
          {isOpen && "Tutorial"}
        </Nav.Link>
      </Nav>
    </div>
  );
}

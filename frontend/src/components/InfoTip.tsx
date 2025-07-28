// src/components/InfoTip.tsx
import React from "react";
import { OverlayTrigger, Tooltip, Button } from "react-bootstrap";
import { InfoCircle } from "react-bootstrap-icons";

export interface InfoTipProps {
  /** The help text shown inside the tooltip */
  text: string;
  /** Optional placement: "top" | "right" | "bottom" | "left" */
  placement?: "top" | "right" | "bottom" | "left";
  /** Optional unique id for accessibility */
  id?: string;
}

const InfoTip: React.FC<InfoTipProps> = ({
  text,
  placement = "right",
  id = "info-tip",
}) => {
  return (
    <OverlayTrigger
      placement={placement}
      overlay={<Tooltip id={id}>{text}</Tooltip>}
    >
      <Button
        variant="link"
        size="sm"
        className="p-0 ms-2 mb-2"
        aria-label="More info"
      >
        <InfoCircle />
      </Button>
    </OverlayTrigger>
  );
};

export default InfoTip;

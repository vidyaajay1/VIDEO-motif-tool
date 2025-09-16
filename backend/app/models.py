from pydantic import BaseModel
from typing import List, Dict, Any

class DatasetInfo(BaseModel):
    data_id: str
    label: str
    peak_list: List[str]

class CompareInitResponse(BaseModel):
    session_id: str
    datasets: List[DatasetInfo]

class BatchScanResponse(BaseModel):
    session_id: str
    datasets: List[str]  # data_ids scanned

class ComparePlotResponse(BaseModel):
    session_id: str
    figures: Dict[str, str]
    ordered_peaks: Dict[str, List[str]]
    final_hits: Dict[str, List[Dict[str, Any]]]  # {label: [peak_ids]}
    
class OverviewResponse(BaseModel):
    genome: str
    peaks_df: List[Dict[str, Any]]
    data_id: str
    peak_list: List[str]

# --- Update your response model to carry Plotly JSON ---
class PlotOverviewResponse(BaseModel):
    data_id: str
    peak_list: List[str]          # ordered to match the plot
    overview_plot: str            # Plotly figure as JSON string
    final_hits: List[Dict[str, Any]] 

class ScannerResponse(BaseModel):
    data_id: str
# tiny status model
class EnqueueResponse(BaseModel):
    job_id: str

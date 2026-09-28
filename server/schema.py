"""Pydantic mirror of src/types/inference.ts. Keep the two in sync."""
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

NormPoint = Tuple[float, float]


class ToothDetection(BaseModel):
    fdi: int = Field(..., description="FDI two-digit tooth number, e.g. 36")
    bbox: Tuple[float, float, float, float] = Field(..., description="x, y, w, h normalised to 0..1")
    polygon: List[NormPoint] = Field(default_factory=list, description="Outline, normalised")
    occlusal: NormPoint = Field(..., description="Centre of the occlusal/incisal edge, normalised")
    cej: Optional[NormPoint] = None
    apex: Optional[NormPoint] = None
    score: Optional[float] = Field(None, ge=0, le=1, description="Model confidence. Only set from real model output.")


class ImageMeta(BaseModel):
    width: int
    height: int
    modality: Literal["OPG"] = "OPG"
    synthetic: bool = False
    fileName: Optional[str] = None


class StageReport(BaseModel):
    id: Literal["preprocess", "detect", "number", "segment", "map", "visualize"]
    label: str
    method: str
    status: Literal["done", "skipped", "failed"] = "done"


class OcclusalCurve(BaseModel):
    a: float
    b: float
    c: float


class InferenceResult(BaseModel):
    schemaVersion: Literal["1.0"] = "1.0"
    source: Literal["model"] = "model"
    notes: Optional[str] = None
    image: ImageMeta
    occlusalCurve: Optional[OcclusalCurve] = None
    teeth: List[ToothDetection]
    stages: List[StageReport]

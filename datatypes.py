from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class Detection:
    """A single bounding box found in a single frame."""
    bbox: tuple
    crop: np.ndarray
    score: float = 0.0          # Default to 0.0 if not provided
    detector_label: int = 0     # Default to 0 (Unknown)

@dataclass
class TrackedVehicle:
    """A vehicle tracked across multiple frames over time."""
    track_id: int
    centroid: tuple
    crops: list = field(default_factory=list) # Automatically creates an empty list
    missing_count: int = 0
    
    # Phase 4 Previews! (Ready for when you need them)
    # direction: str = "unknown"
    # counted: bool = False
    # timestamp: float = field(default_factory=time.time)

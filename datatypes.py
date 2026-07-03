from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class Detection:
    """A single bounding box found in a single frame."""
    bbox: tuple
    crop: np.ndarray
    score: float = 0.0
    detector_label: int = 0


@dataclass
class TrackedVehicle:
    """A vehicle tracked across multiple frames over time."""
    track_id: int
    centroid: tuple
    crops: list = field(default_factory=list)
    missing_count: int = 0

    # Most common detector label seen across all frames for this vehicle.
    # Stored as a list of raw per-frame guesses so we can majority-vote it,
    # same way the classifier votes across frames.
    detector_labels: list = field(default_factory=list)

    # Phase 4 Previews! (Ready for when you need them)
    # direction: str = "unknown"
    # counted: bool = False
    # timestamp: float = field(default_factory=time.time)

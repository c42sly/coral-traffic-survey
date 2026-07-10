import math
from collections import OrderedDict
from datatypes import TrackedVehicle


class SmartBufferTracker:
    def __init__(self, max_distance=150, max_frames=20, max_missing=5):
        self.active_tracks = OrderedDict()  # track_id -> TrackedVehicle object
        self.next_track_id = 0

        self.max_distance = max_distance
        self.max_frames = max_frames
        self.max_missing = max_missing

    def update(self, detections, current_time):
        """
        detections: list of Detection dataclass objects
        Returns: (completed_tracks, discarded_tracks) — both lists of TrackedVehicle.
        completed_tracks have enough crops (>2) to send to the classifier.
        discarded_tracks timed out with too few crops (likely flicker/false positive,
        or a real vehicle that fragmented into a new track id) — kept only for
        debug inspection, never sent to the classifier.
        """
        completed_tracks = []

        # 1. Update missed frames for all existing tracks
        for track_id, vehicle in self.active_tracks.items():
            vehicle.missing_count += 1

        # 2. Match new detections to existing tracks (exclusive, globally-greedy)
        #
        # Each detection used to pick its own nearest track independently, which let
        # two different detections in the same frame both claim the same track (and
        # have their crops blended into it) while another track went unmatched and
        # spawned a duplicate. Instead, we now gather every valid (detection, track)
        # pair within max_distance, sort by distance, and assign in that order —
        # once a detection or a track is claimed, neither can be matched again this
        # frame. This is a simple greedy approximation of optimal assignment
        # (not a full min-cost solve like the Hungarian algorithm), but it's enough
        # to guarantee a 1:1 detection-to-track mapping per frame.

        # Precompute each detection's centroid once.
        det_centroids = []
        for det in detections:
            xmin, ymin, xmax, ymax = det.bbox
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
            det_centroids.append((cx, cy))

        # Build all candidate (distance, det_idx, track_id) triples within range.
        candidates = []
        for det_idx, (cx, cy) in enumerate(det_centroids):
            for track_id, vehicle in self.active_tracks.items():
                old_cx, old_cy = vehicle.centroid
                dist = math.hypot(cx - old_cx, cy - old_cy)
                if dist < self.max_distance:
                    candidates.append((dist, det_idx, track_id))

        # Closest pairs win first.
        candidates.sort(key=lambda c: c[0])

        matched_det_idxs = set()
        matched_track_ids = set()

        for dist, det_idx, track_id in candidates:
            if det_idx in matched_det_idxs or track_id in matched_track_ids:
                continue  # detection or track already claimed this frame

            cx, cy = det_centroids[det_idx]
            vehicle = self.active_tracks[track_id]
            vehicle.centroid = (cx, cy)
            vehicle.crops.append(detections[det_idx].crop)
            vehicle.missing_count = 0

            # Accumulate the detector's per-frame label guess alongside the crop,
            # so we can majority-vote it later in the classifier, same as crops.
            vehicle.detector_labels.append(detections[det_idx].detector_label)

            matched_det_idxs.add(det_idx)
            matched_track_ids.add(track_id)

        # Any detection that didn't get claimed becomes a new track.
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_idxs:
                continue
            cx, cy = det_centroids[det_idx]
            self.active_tracks[self.next_track_id] = TrackedVehicle(
                track_id=self.next_track_id,
                centroid=(cx, cy),
                crops=[det.crop],
                missing_count=0,
                # Seed detector_labels with the first frame's guess
                detector_labels=[det.detector_label],
            )
            self.next_track_id += 1

        # 3. Check for Triggers
        ready_ids = []
        discarded_tracks = []
        for track_id, vehicle in self.active_tracks.items():
            if len(vehicle.crops) >= self.max_frames or vehicle.missing_count > self.max_missing:
                if len(vehicle.crops) > 2:
                    completed_tracks.append(vehicle)
                else:
                    # Too few crops to bother classifying. Could be flicker/a false
                    # positive, or a real vehicle that got re-matched under a *new*
                    # track id after moving further than max_distance in one frame.
                    # Surface it (with its crops) instead of silently dropping it,
                    # so it can actually be inspected.
                    discarded_tracks.append(vehicle)
                    print(f"🗑️  Track {track_id} discarded: {len(vehicle.crops)} crop(s), "
                          f"missing {vehicle.missing_count}/{self.max_missing} frames, "
                          f"last centroid {vehicle.centroid}, "
                          f"detector guesses: {vehicle.detector_labels}")
                ready_ids.append(track_id)

        # Remove from tracker memory
        for track_id in ready_ids:
            del self.active_tracks[track_id]

        return completed_tracks, discarded_tracks
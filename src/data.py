from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


CLASSES = ("pinky", "elle", "yo", "index", "thumb")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
ANNOTATION_HZ = 200.0
SENSOR_SIZE = 128


@dataclass(frozen=True)
class Segment:
    label: str
    start_s: float
    end_s: float
    repetition: int


def annotation_segments(labels: np.ndarray) -> list[Segment]:
    """Convert the 200 Hz annotation vector into contiguous gesture segments."""
    if labels.ndim != 1:
        raise ValueError(f"Expected a 1-D annotation array, got {labels.shape}")

    segments: list[Segment] = []
    repetitions = {name: 0 for name in CLASSES}
    active_label = "none"
    active_start = 0

    for idx, value in enumerate(labels.astype(str)):
        if value == active_label:
            continue
        if active_label != "none":
            repetitions[active_label] += 1
            segments.append(
                Segment(
                    active_label,
                    active_start / ANNOTATION_HZ,
                    idx / ANNOTATION_HZ,
                    repetitions[active_label],
                )
            )
        active_label = value
        active_start = idx

    if active_label != "none":
        repetitions[active_label] += 1
        segments.append(
            Segment(
                active_label,
                active_start / ANNOTATION_HZ,
                len(labels) / ANNOTATION_HZ,
                repetitions[active_label],
            )
        )

    invalid = sorted({segment.label for segment in segments} - set(CLASSES))
    if invalid:
        raise ValueError(f"Unexpected gesture labels: {invalid}")
    return segments


def frame_segment(
    events: np.ndarray,
    segment: Segment,
    timesteps: int = 8,
    spatial_size: int = 32,
) -> tuple[np.ndarray, int]:
    """Bin an asynchronous DVS segment into equal-duration polarity frames."""
    if events.ndim != 2 or events.shape[0] != 4:
        raise ValueError(f"Expected events with shape (4, N), got {events.shape}")
    if segment.end_s <= segment.start_s:
        raise ValueError("Segment end must be after start")

    timestamps = events[2]
    selected = (timestamps >= segment.start_s) & (timestamps < segment.end_s)
    x = events[0, selected].astype(np.int64, copy=False)
    y = events[1, selected].astype(np.int64, copy=False)
    t = timestamps[selected]
    polarity = events[3, selected].astype(np.int64, copy=False)

    if x.size == 0:
        raise ValueError(f"No events found for {segment}")
    if x.min() < 0 or x.max() >= SENSOR_SIZE or y.min() < 0 or y.max() >= SENSOR_SIZE:
        raise ValueError("DVS coordinates are outside the expected 128x128 sensor bounds")
    if not np.isin(polarity, (0, 1)).all():
        raise ValueError("Polarity must be binary")

    time_bin = np.floor(
        (t - segment.start_s) * timesteps / (segment.end_s - segment.start_s)
    ).astype(np.int64)
    time_bin = np.clip(time_bin, 0, timesteps - 1)
    x_bin = np.minimum(x * spatial_size // SENSOR_SIZE, spatial_size - 1)
    y_bin = np.minimum(y * spatial_size // SENSOR_SIZE, spatial_size - 1)

    flat = (((time_bin * 2 + polarity) * spatial_size + y_bin) * spatial_size + x_bin)
    counts = np.bincount(flat, minlength=timesteps * 2 * spatial_size * spatial_size)
    frames = counts.reshape(timesteps, 2, spatial_size, spatial_size).astype(np.float32)

    # Log compression preserves sparsity while reducing the influence of hot pixels.
    frames = np.log1p(frames)
    scale = float(frames.max())
    if scale > 0:
        frames /= scale
    return frames, int(x.size)


def prepare_smoke_dataset(
    raw_dir: Path,
    output_path: Path,
    sessions: Iterable[str] = (
        "subject03_session03",
        "subject05_session01",
        "subject10_session01",
        "subject07_session03",
    ),
    timesteps: int = 8,
    spatial_size: int = 32,
) -> dict:
    frames_all: list[np.ndarray] = []
    labels_all: list[int] = []
    session_all: list[str] = []
    repetition_all: list[int] = []
    event_counts: list[int] = []
    starts: list[float] = []
    ends: list[float] = []

    for session in sessions:
        annotation_path = raw_dir / f"{session}_ann.npy"
        event_path = raw_dir / f"{session}_dvs.npy"
        annotations = np.load(annotation_path, allow_pickle=False)
        events = np.load(event_path, mmap_mode="r", allow_pickle=False)
        segments = annotation_segments(annotations)

        if len(segments) != 25:
            raise ValueError(f"{session}: expected 25 gesture segments, got {len(segments)}")
        for name in CLASSES:
            if sum(segment.label == name for segment in segments) != 5:
                raise ValueError(f"{session}: expected 5 repetitions for {name}")

        for segment in segments:
            frames, count = frame_segment(events, segment, timesteps, spatial_size)
            frames_all.append(frames)
            labels_all.append(CLASS_TO_ID[segment.label])
            session_all.append(session)
            repetition_all.append(segment.repetition)
            event_counts.append(count)
            starts.append(segment.start_s)
            ends.append(segment.end_s)

    # uint8 keeps the derived cache below 1 MiB without changing the raw source files.
    x_uint8 = np.rint(np.stack(frames_all) * 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x_uint8,
        y=np.asarray(labels_all, dtype=np.int64),
        session=np.asarray(session_all),
        repetition=np.asarray(repetition_all, dtype=np.int64),
        event_count=np.asarray(event_counts, dtype=np.int64),
        start_s=np.asarray(starts, dtype=np.float64),
        end_s=np.asarray(ends, dtype=np.float64),
        classes=np.asarray(CLASSES),
    )

    return {
        "samples": len(labels_all),
        "shape": list(x_uint8.shape),
        "class_counts": {name: int(labels_all.count(CLASS_TO_ID[name])) for name in CLASSES},
        "sessions": list(sessions),
        "events_total": int(sum(event_counts)),
        "events_min": int(min(event_counts)),
        "events_max": int(max(event_counts)),
        "events_mean": float(np.mean(event_counts)),
        "duration_min_s": float(np.min(np.asarray(ends) - np.asarray(starts))),
        "duration_max_s": float(np.max(np.asarray(ends) - np.asarray(starts))),
    }


class GestureSmokeDataset(Dataset):
    def __init__(self, processed_path: Path, split: str, augment: bool = False):
        data = np.load(processed_path, allow_pickle=False)
        x = data["x"]
        y = data["y"]
        sessions = data["session"].astype(str)
        repetitions = data["repetition"]

        training_subject = sessions != "subject07_session03"
        if split == "train":
            mask = training_subject & (repetitions <= 4)
        elif split == "val":
            mask = training_subject & (repetitions == 5)
        elif split == "test":
            mask = sessions == "subject07_session03"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.x = torch.from_numpy(x[mask].astype(np.float32) / 255.0)
        self.y = torch.from_numpy(y[mask].astype(np.int64))
        self.augment = augment

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self.x[index]
        if self.augment:
            shift_y = int(torch.randint(-2, 3, ()).item())
            shift_x = int(torch.randint(-2, 3, ()).item())
            shifted = torch.roll(frames, shifts=(shift_y, shift_x), dims=(-2, -1))
            if shift_y > 0:
                shifted[..., :shift_y, :] = 0
            elif shift_y < 0:
                shifted[..., shift_y:, :] = 0
            if shift_x > 0:
                shifted[..., :, :shift_x] = 0
            elif shift_x < 0:
                shifted[..., :, shift_x:] = 0
            keep = (torch.rand_like(shifted) > 0.03).to(shifted.dtype)
            scale = 0.90 + 0.20 * torch.rand(())
            frames = (shifted * keep * scale).clamp_(0.0, 1.0)
        return frames, self.y[index]

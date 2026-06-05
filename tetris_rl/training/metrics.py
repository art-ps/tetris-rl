"""Training metric records and CSV persistence."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    episode: int
    reward: float
    score: int
    lines: int
    pieces: int
    epsilon: float
    loss: float
    avg_reward_100: float
    avg_lines_100: float


class MetricsWriter:
    """Append episode records to one CSV file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = list(EpisodeMetrics.__dataclass_fields__)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as output:
                csv.DictWriter(output, fieldnames=self._fieldnames).writeheader()

    def append(self, metrics: EpisodeMetrics) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=self._fieldnames)
            writer.writerow(asdict(metrics))

import csv

from tetris_rl.training.metrics import EpisodeMetrics, MetricsWriter


def test_metrics_writer_creates_csv_and_appends_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    writer = MetricsWriter(path)
    writer.append(
        EpisodeMetrics(
            episode=1,
            reward=2.5,
            score=100,
            lines=1,
            pieces=10,
            epsilon=0.9,
            loss=0.25,
            avg_reward_100=2.5,
            avg_lines_100=1,
        )
    )

    with path.open() as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    assert len(rows) == 1
    assert rows[0]["episode"] == "1"
    assert rows[0]["score"] == "100"

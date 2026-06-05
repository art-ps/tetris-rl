import csv

from scripts.plot_metrics import load_metrics, plot_run


def test_plot_run_creates_three_images(tmp_path):
    path = tmp_path / "metrics.csv"
    fieldnames = [
        "episode",
        "reward",
        "score",
        "lines",
        "pieces",
        "epsilon",
        "loss",
        "avg_reward_100",
        "avg_lines_100",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict.fromkeys(fieldnames, 1))

    metrics = load_metrics(tmp_path)
    outputs = plot_run(tmp_path)

    assert metrics["episode"] == [1.0]
    assert {path.name for path in outputs} == {"reward.png", "lines.png", "loss.png"}
    assert all(path.exists() for path in outputs)

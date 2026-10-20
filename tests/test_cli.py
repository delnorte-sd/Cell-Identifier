from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cell_identifier import __version__
from cell_identifier.cli import main


@pytest.fixture
def synthetic_dir(tmp_path: Path) -> Path:
    out = tmp_path / "synthetic"
    assert main(["generate", "-o", str(out), "--seed", "1", "-n", "25", "--count", "2"]) == 0
    return out


def test_generate_writes_image_labels_and_ground_truth(synthetic_dir: Path) -> None:
    names = sorted(p.name for p in synthetic_dir.iterdir())
    assert names == [
        "synthetic_000.png",
        "synthetic_000_ground_truth.json",
        "synthetic_000_labels.png",
        "synthetic_001.png",
        "synthetic_001_ground_truth.json",
        "synthetic_001_labels.png",
    ]
    gt = json.loads((synthetic_dir / "synthetic_001_ground_truth.json").read_text(encoding="utf-8"))
    assert gt["placed_cells"] == 25
    assert gt["config"]["seed"] == 2


def test_analyze_directory_batch(synthetic_dir: Path, tmp_path: Path, capsys) -> None:
    (synthetic_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    out = tmp_path / "results"
    code = main(
        [
            "analyze",
            str(synthetic_dir / "synthetic_000.png"),
            str(synthetic_dir / "synthetic_001.png"),
            "-o",
            str(out),
            "--set",
            "filter.min_area=10",
        ]
    )
    assert code == 0
    assert "25 cells" in capsys.readouterr().out
    for stem in ("synthetic_000", "synthetic_001"):
        report = json.loads((out / stem / "results.json").read_text(encoding="utf-8"))
        assert report["summary"]["cell_count"] == 25
        assert report["config"]["filter"]["min_area"] == 10
        assert (out / stem / "overlay.png").is_file()
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["status"] for r in rows] == ["ok", "ok"]


def test_analyze_folder_with_config_file(synthetic_dir: Path, tmp_path: Path) -> None:
    config = tmp_path / "analysis.toml"
    config.write_text('[classification]\nmethod = "none"\n', encoding="utf-8")
    out = tmp_path / "res"
    assert (
        main(["analyze", str(synthetic_dir), "-o", str(out), "-c", str(config), "--no-images"]) == 0
    )
    # The folder also contains the *_labels.png ground-truth images; all are analysed.
    assert len(list(out.glob("*/results.json"))) == 4
    assert not list(out.glob("*/overlay.png"))
    report = json.loads((out / "synthetic_000" / "results.json").read_text(encoding="utf-8"))
    assert {c["brightness"] for c in report["cells"]} == {"Unclassified"}


def test_analyze_reports_failures_but_continues(
    synthetic_dir: Path, tmp_path: Path, capsys
) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    out = tmp_path / "res"
    code = main(["analyze", str(broken), str(synthetic_dir / "synthetic_000.png"), "-o", str(out)])
    assert code == 1
    assert "FAILED" in capsys.readouterr().err
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        assert [r["status"] for r in csv.DictReader(handle)] == ["failed", "ok"]


def test_analyze_usage_errors(tmp_path: Path, capsys) -> None:
    assert main(["analyze", str(tmp_path), "-o", str(tmp_path / "o")]) == 2  # empty dir
    img = tmp_path / "x.png"
    from cell_identifier.image_io import save_image

    save_image(img, __import__("numpy").zeros((16, 16), dtype="uint8"))
    assert main(["analyze", str(img), "--set", "filter.min_area=-5"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_benchmark(tmp_path: Path, capsys) -> None:
    out = tmp_path / "bench.json"
    assert main(["benchmark", "--count", "2", "--seed", "3", "-n", "20", "--json", str(out)]) == 0
    assert "mean F1 = 1.0000" in capsys.readouterr().out
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [r["seed"] for r in rows] == [3, 4]


def test_config_command(capsys) -> None:
    assert main(["config", "--set", "segmentation.min_snr=2"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["segmentation"]["min_snr"] == 2.0
    assert main(["config", "--synthetic"]) == 0
    assert "num_cells" in json.loads(capsys.readouterr().out)


def test_version_and_module_entry_point() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    completed = subprocess.run(
        [sys.executable, "-m", "cell_identifier", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in completed.stdout

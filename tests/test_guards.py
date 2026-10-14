"""Guard tests for the acceptance criteria. Each docstring is Given / When / Then."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hidrometros import criteria, data
from hidrometros.models import MODEL_NAMES

needs_digits = pytest.mark.skipif(not (data.DIGITS_DIR / "index.csv").exists(), reason="digits/ not generated locally")


def test_c01_splits_sha_matches_professor() -> None:
    """Given the repo splits.csv, When hashed, Then it equals the professor's sha256."""
    assert criteria.c01_splits_sha().ok


def test_c01_detects_altered_splits(tmp_path: Path) -> None:
    """Given a splits.csv with one row changed, When checked, Then the guard fails."""
    altered = tmp_path / "splits.csv"
    altered.write_text(data.SPLITS_CSV.read_text().replace("train", "val", 1))
    assert not criteria.c01_splits_sha(altered).ok


@needs_digits
def test_c02_counts_and_index_columns() -> None:
    """Given the generated digits/index.csv, When counted, Then 1243 valid, 1 discarded, 9944 crops."""
    assert criteria.c02_counts().ok


@needs_digits
def test_c03_no_image_in_two_splits() -> None:
    """Given digits/index.csv, When grouped by photo_name, Then no image appears in two splits."""
    assert criteria.c03_no_leak().ok


@pytest.mark.parametrize("n_leaks", [1, 3])
def test_c03_detects_leakage(tmp_path: Path, n_leaks: int) -> None:
    """Given an index where n images also appear in another split, When checked, Then they are reported."""
    rows = [{"photo_name": f"p{i}", "split": "train"} for i in range(5)]
    rows += [{"photo_name": f"p{i}", "split": "test"} for i in range(n_leaks)]
    assert len(criteria.leaking_images(pd.DataFrame(rows))) == n_leaks
    p = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert not criteria.c03_no_leak(p).ok


def test_c05_protocol_table_and_no_flip() -> None:
    """Given the code, When the protocol is inspected, Then it equals the assignment table with no flip."""
    res = criteria.c05_protocol(history_dir=Path("/nonexistent"))
    assert "flip" not in res.detail and "!=" not in res.detail  # history files absent is checked in the real run


def test_c07_detects_wrong_results_shape(tmp_path: Path) -> None:
    """Given a results.csv with 3 rows and a missing column, When checked, Then the guard fails."""
    bad = tmp_path / "results.csv"
    pd.DataFrame({"model": ["smallcnn", "alexnet", "resnet18"], "acc_digit": [0.9, 0.9, 0.9]}).to_csv(bad, index=False)
    assert not criteria.c07_results(bad, tmp_path).ok


def test_c07_detects_validation_metrics(tmp_path: Path) -> None:
    """Given predictions that include a non-test image, When checked, Then the guard fails."""
    reports = tmp_path
    for name in MODEL_NAMES:
        pd.DataFrame({"photo_name": ["id_1_value_13_116.jpg"], "y_true": [0], "y_pred": [0]}).to_csv(
            reports / f"predictions_{name}.csv", index=False
        )
    res = tmp_path / "results.csv"
    pd.DataFrame([[m, 1, 1, 1, 1, 1, 1, 1, 1] for m in MODEL_NAMES], columns=criteria.RESULT_COLUMNS).to_csv(res, index=False)
    out = criteria.c07_results(res, reports)
    assert not out.ok and "not exactly the test split" in out.detail


def test_c07_real_results() -> None:
    """Given the committed results.csv, When recomputed from test predictions, Then it matches."""
    if not (data.ROOT / "results.csv").exists():
        pytest.skip("results.csv not generated yet")
    assert criteria.c07_results().ok


def _nb(path: Path, mime: str | None) -> Path:
    outputs = [{"output_type": "display_data", "data": {mime: "AAAA"}, "metadata": {}}] if mime else []
    nb = {"cells": [{"cell_type": "code", "source": [], "outputs": outputs, "metadata": {}, "execution_count": 1}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb))
    return path


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
def test_c10_detects_embedded_images(tmp_path: Path, mime: str) -> None:
    """Given a notebook with an image output, When scanned, Then it is counted and the guard fails."""
    p = _nb(tmp_path / "n.ipynb", mime)
    assert criteria.notebook_images(p) == 1
    assert not criteria.c10_notebook(p).ok


def test_c10_clean_notebook_passes(tmp_path: Path) -> None:
    """Given a notebook with no image outputs, When scanned, Then the guard passes."""
    assert criteria.c10_notebook(_nb(tmp_path / "n.ipynb", None)).ok


def test_c10_real_notebook() -> None:
    """Given the committed notebook, When scanned, Then it has no image outputs."""
    p = data.ROOT / "notebooks" / "pratica_watermeters.ipynb"
    if not p.exists():
        pytest.skip("notebook not written yet")
    assert criteria.c10_notebook(p).ok


def _readme(tmp: Path, table_acc: str = "0.9000", drop: str | None = None) -> tuple[Path, Path]:
    res = tmp / "results.csv"
    pd.DataFrame([[m, "0.9000", "0.9000", "0.5000", "1.00", "1.0", "1.00", "1.0", "3"] for m in MODEL_NAMES], columns=criteria.RESULT_COLUMNS).to_csv(res, index=False)
    lines = ["| model | acc |", *[f"| {m} | {table_acc if m == 'smallcnn' else '0.9000'} | 0.9000 | 0.5000 | 1.00 | 1.0 | 1.00 | 1.0 | 3 |" for m in MODEL_NAMES]]
    answers = "\n".join(f"**{q}** " + "texto de resposta suficientemente longo para passar no limite minimo." for q in criteria.QUESTIONS if q != drop)
    text = "\n".join(
        [" ".join(criteria.MEMBER_EMAILS), "## Como reproduzir", "## Protocolo de treino", "## Limitações", "## Declaração de uso de IA", "docs/SPEC.md", *lines, "## Respostas", answers]
    )
    p = tmp / "README.md"
    p.write_text(text)
    return p, res


def test_c11_complete_readme_passes(tmp_path: Path) -> None:
    """Given a README with members, sections, 17 answers and a table equal to results.csv, Then it passes."""
    readme, res = _readme(tmp_path)
    assert criteria.c11_readme(readme, res).ok


def test_c11_detects_table_mismatch(tmp_path: Path) -> None:
    """Given a README table that differs from results.csv in one cell, When checked, Then it fails."""
    readme, res = _readme(tmp_path, table_acc="0.9001")
    out = criteria.c11_readme(readme, res)
    assert not out.ok and "results table" in out.detail


def test_c11_detects_missing_answer(tmp_path: Path) -> None:
    """Given a README without answer 4.4, When checked, Then it fails."""
    readme, res = _readme(tmp_path, drop="4.4")
    out = criteria.c11_readme(readme, res)
    assert not out.ok and "answer 4.4" in out.detail


def test_c11_real_readme() -> None:
    """Given the committed README, When checked against results.csv, Then it passes."""
    if not (data.ROOT / "README.md").exists():
        pytest.skip("README not written yet")
    out = criteria.c11_readme()
    assert out.ok, out.detail


@pytest.mark.parametrize(
    "path",
    ["data/WaterMeters/images/a.jpg", "digits/train/0/1_0.png", "checkpoints/x/best.pt", "weights.pth", "m.ckpt", "kaggle.json", "~/x/credentials.json", "notebooks/leak.png", "src/a.JPG", "local/audit/a.txt"],
)
def test_c12_flags_forbidden_paths(path: str) -> None:
    """Given a tracked path of a forbidden kind, When classified, Then it is flagged."""
    assert criteria.forbidden_tracked([path]) == [path]


@pytest.mark.parametrize("path", ["figures/confusion_matrices.png", "docs/SPEC.md", "src/hidrometros/data.py", "splits.csv", "data_overrides.csv", "reports/predictions_resnet18.csv"])
def test_c12_allows_normal_paths(path: str) -> None:
    """Given a normal tracked path, When classified, Then it is not flagged."""
    assert criteria.forbidden_tracked([path]) == []


def test_c12_real_git_index() -> None:
    """Given the real git index, When listed, Then no forbidden path or non-plot figure is tracked."""
    out = criteria.c12_git_clean()
    assert out.ok, out.detail


def test_figures_are_matplotlib_plots(tmp_path: Path) -> None:
    """Given a tracked PNG that was not written by matplotlib, When checked, Then it is flagged as non-plot."""
    from PIL import Image

    (tmp_path / "figures").mkdir()
    Image.new("RGB", (10, 10)).save(tmp_path / "figures" / "photo.png")
    assert criteria.non_plot_figures(["figures/photo.png"], tmp_path) == ["figures/photo.png"]


def test_label_parsing_padding() -> None:
    """Given filenames with short integer or decimal parts, When parsed, Then they pad to 5+3 digits."""
    assert data.label_from_name("id_1_value_13_116.jpg") == "00013116"
    assert data.label_from_name("id_2_value_21_86.jpg") == "00021860"
    assert data.label_from_name("id_664_value_45_2676.jpg") is None


def test_real_matplotlib_png_is_accepted(tmp_path: Path) -> None:
    """Given a PNG written by matplotlib (Software tag 'Matplotlib version...'), When checked, Then it is a plot."""
    import matplotlib.pyplot as plt

    (tmp_path / "figures").mkdir()
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    fig.savefig(tmp_path / "figures" / "plot.png")
    plt.close(fig)
    assert criteria.non_plot_figures(["figures/plot.png"], tmp_path) == []

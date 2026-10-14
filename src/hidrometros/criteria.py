"""Acceptance criteria 1-13 (assignment + spec section 9). Owner of every check.

`hidrometros.check` prints them; `tests/` assert them and also feed them deliberately broken inputs.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from hidrometros import data
from hidrometros.models import MODEL_NAMES, PROTOCOL

ROOT = data.ROOT
INDEX_COLUMNS = ["image_id", "photo_name", "position", "digit", "split", "crop_path"]
HISTORY_COLUMNS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "seconds"]
RESULT_COLUMNS = [
    "model", "acc_digit", "f1_macro", "acc_reading", "params_M",
    "s_per_epoch", "ms_per_image", "ckpt_MB", "best_epoch",
]  # fmt: skip
FIGURES = ["confusion_matrices.png", "accuracy_by_position.png", "training_curves.png", "performance_vs_cost.png"]
QUESTIONS = ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "3.3", "4.1", "4.2", "4.3", "4.4", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6"]
MEMBER_EMAILS = [
    "ads2@cesar.school", "cam3@cesar.school", "fcrs@cesar.school",
    "gbr2@cesar.school", "icaj@cesar.school", "mam2@cesar.school", "jhlr@cesar.school",
]  # fmt: skip
EXPECTED = {"valid": 1243, "discarded": 1, "crops": 9944}


@dataclass(frozen=True)
class Result:
    ok: bool
    detail: str


def _ok(cond: bool, detail: str) -> Result:
    return Result(cond, detail)


# 1 ---------------------------------------------------------------------------------------------
def c01_splits_sha(path: Path = data.SPLITS_CSV) -> Result:
    sha = data.sha256_of(path)
    return _ok(sha == data.SPLITS_SHA256, f"sha256 {sha[:12]}...")


# 2 ---------------------------------------------------------------------------------------------
def c02_counts(index_csv: Path = data.DIGITS_DIR / "index.csv") -> Result:
    if not index_csv.exists():
        return Result(False, f"{index_csv} missing (run `hidrometros prep`)")
    idx = pd.read_csv(index_csv)
    ann = data.load_annotations()
    n_valid = int((ann["label"].notna() & ann["split"].notna()).sum())
    n_disc = int(len(ann) - n_valid)
    cols_ok = list(idx.columns) == INDEX_COLUMNS
    ok = (n_valid, n_disc, len(idx)) == (EXPECTED["valid"], EXPECTED["discarded"], EXPECTED["crops"]) and cols_ok
    return _ok(ok, f"valid={n_valid} discarded={n_disc} crops={len(idx)} columns_ok={cols_ok}")


# 3 ---------------------------------------------------------------------------------------------
def leaking_images(index: pd.DataFrame) -> list[str]:
    """photo_names that appear in more than one split."""
    n = index.groupby("photo_name")["split"].nunique()
    return n[n > 1].index.tolist()


def c03_no_leak(index_csv: Path = data.DIGITS_DIR / "index.csv") -> Result:
    if not index_csv.exists():
        return Result(False, f"{index_csv} missing")
    idx = pd.read_csv(index_csv)
    splits = pd.read_csv(data.SPLITS_CSV).set_index("photo_name")["split"]
    mismatched = int((idx["split"] != idx["photo_name"].map(splits)).sum())
    leaks = leaking_images(idx)
    return _ok(not leaks and mismatched == 0, f"images in >1 split={len(leaks)} crops disagreeing with splits.csv={mismatched}")


# 4 ---------------------------------------------------------------------------------------------
def c04_orientation_reported(readme: Path = ROOT / "README.md", stats: Path = ROOT / "reports" / "prep_stats.json") -> Result:
    if not readme.exists() or not stats.exists():
        return Result(False, "README.md or reports/prep_stats.json missing")
    st = json.loads(stats.read_text())
    ans = _answer(readme.read_text(), "3.2")
    needed = [str(st["rotated_by_geometry"]), str(st["flipped_final"])]
    missing = [n for n in needed if not re.search(rf"(?<!\d){n}(?!\d)", ans)]
    return _ok(not missing and bool(ans), f"answer 3.2 cites {needed}; missing={missing}")


# 5 ---------------------------------------------------------------------------------------------
_SPEC_TABLE = {
    "smallcnn": (False, (64, 48), 256, 128, 1e-3, None, 10),
    "alexnet": (True, (128, 128), 128, 64, 1e-3, 1e-4, 8),
    "vgg16_bn": (True, (128, 128), 32, 16, 1e-3, 1e-5, 8),
    "resnet18": (True, (128, 128), 128, 64, 1e-3, 1e-4, 8),
}


def c05_protocol(history_dir: Path = ROOT) -> Result:
    from torchvision.transforms import v2

    from hidrometros.models import EARLY_STOP_PATIENCE, LABEL_SMOOTHING, SEED
    from hidrometros.train import train_transform

    problems: list[str] = []
    for name, want in _SPEC_TABLE.items():
        s = PROTOCOL[name]
        got = (s.pretrained, s.input_hw, s.batch_cuda, s.batch_mps_cpu, s.lr_head, s.lr_backbone, s.max_epochs)
        if got != want:
            problems.append(f"{name}: {got} != {want}")
        tf = train_transform(s)
        if any(isinstance(t, (v2.RandomHorizontalFlip, v2.RandomVerticalFlip)) for t in tf.transforms):
            problems.append(f"{name}: flip in augmentation")
        hp = history_dir / f"history_{name}.csv"
        if not hp.exists():
            problems.append(f"{name}: {hp.name} missing")
        elif len(pd.read_csv(hp)) > s.max_epochs:
            problems.append(f"{name}: more epochs than max")
    if (SEED, EARLY_STOP_PATIENCE, LABEL_SMOOTHING) != (42, 3, 0.1):
        problems.append("seed/patience/label smoothing differ from protocol")
    return _ok(not problems, "; ".join(problems) or "4 models match the protocol table; no flip; seed 42")


# 6 ---------------------------------------------------------------------------------------------
def c06_history_columns(history_dir: Path = ROOT) -> Result:
    bad = []
    for name in MODEL_NAMES:
        p = history_dir / f"history_{name}.csv"
        if not p.exists() or list(pd.read_csv(p).columns) != HISTORY_COLUMNS:
            bad.append(name)
    return _ok(not bad, f"bad or missing: {bad}" if bad else "4 history files with the 7 columns")


# 7 ---------------------------------------------------------------------------------------------
def c07_results(results_csv: Path = ROOT / "results.csv", reports_dir: Path = ROOT / "reports") -> Result:
    if not results_csv.exists():
        return Result(False, "results.csv missing")
    df = pd.read_csv(results_csv)
    problems: list[str] = []
    if list(df.columns) != RESULT_COLUMNS:
        problems.append(f"columns {list(df.columns)}")
    if sorted(df["model"].tolist()) != sorted(MODEL_NAMES):
        problems.append(f"models {df['model'].tolist()}")
    test_names = set(pd.read_csv(data.SPLITS_CSV).query("split == 'test'")["photo_name"])
    for name in MODEL_NAMES:
        p = reports_dir / f"predictions_{name}.csv"
        if not p.exists():
            problems.append(f"{p.name} missing")
            continue
        pr = pd.read_csv(p)
        if not set(pr["photo_name"]) <= test_names or len(pr) != len(test_names) * data.N_DIGITS:
            problems.append(f"{name}: predictions are not exactly the test split")
        row = df[df["model"] == name]
        if len(row) == 1 and abs(float((pr["y_true"] == pr["y_pred"]).mean()) - float(row["acc_digit"].iloc[0])) > 5e-5:
            problems.append(f"{name}: acc_digit differs from test predictions")
    return _ok(not problems, "; ".join(problems) or "4 rows x 9 columns; recomputed from test-only predictions")


# 8 ---------------------------------------------------------------------------------------------
def c08_figures(figures_dir: Path = ROOT / "figures", readme: Path = ROOT / "README.md") -> Result:
    text = readme.read_text() if readme.exists() else ""
    problems = [f for f in FIGURES if not (figures_dir / f).exists() or (figures_dir / f).stat().st_size < 5000]
    not_embedded = [f for f in FIGURES if f"figures/{f}" not in text or f"![" not in text]
    return _ok(not problems and not not_embedded, f"missing={problems} not_embedded={not_embedded}")


# 9 ---------------------------------------------------------------------------------------------
def c09_reports(reports_dir: Path = ROOT / "reports") -> Result:
    missing = [n for n in MODEL_NAMES if not (reports_dir / f"classification_report_{n}.txt").exists()]
    return _ok(not missing, f"missing: {missing}" if missing else "4 classification reports")


# 10 --------------------------------------------------------------------------------------------
def notebook_images(path: Path) -> int:
    """Number of embedded image outputs (png/jpeg/gif/svg) in a notebook."""
    nb = json.loads(path.read_text())
    n = 0
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            mimes = out.get("data", {}).keys()
            n += sum(m.startswith("image/") for m in mimes)
    return n


def c10_notebook(path: Path = ROOT / "notebooks" / "pratica_watermeters.ipynb") -> Result:
    if not path.exists():
        return Result(False, f"{path.name} missing")
    n = notebook_images(path)
    return _ok(n == 0, f"{n} embedded image outputs")


# 11 --------------------------------------------------------------------------------------------
def _answer(text: str, q: str) -> str:
    m = re.search(rf"\*\*{re.escape(q)}\*\*(.*?)(?=\n\*\*\d\.\d\*\*|\n## |\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def readme_results_block(text: str) -> list[list[str]]:
    """Rows of the results table in the README (header excluded), as strings."""
    lines = [ln for ln in text.splitlines() if ln.startswith("| ") and re.match(r"\|\s*(smallcnn|alexnet|vgg16_bn|resnet18)\b", ln)]
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    return [r for r in rows if len(r) == len(RESULT_COLUMNS)]


def c11_readme(
    readme: Path = ROOT / "README.md", results_csv: Path = ROOT / "results.csv", spec_link: str = "docs/SPEC.md"
) -> Result:
    if not readme.exists():
        return Result(False, "README.md missing")
    text = readme.read_text()
    problems: list[str] = []
    problems += [f"member {e}" for e in MEMBER_EMAILS if e not in text]
    problems += [f"answer {q}" for q in QUESTIONS if len(_answer(text, q)) < 40]
    for section in ("Como reproduzir", "Protocolo de treino", "Limitações", "Declaração de uso de IA"):
        if section not in text:
            problems.append(f"section {section}")
    if spec_link not in text:
        problems.append("link to docs/SPEC.md")
    if re.search(r"^>\s", text, re.M):
        problems.append("instruction lines ('>') not removed")
    if results_csv.exists():
        want = pd.read_csv(results_csv, dtype=str).values.tolist()
        got = readme_results_block(text)
        if got != want:
            problems.append("results table != results.csv")
    else:
        problems.append("results.csv missing")
    return _ok(not problems, "; ".join(problems) or "members, 17 answers, sections, spec link, table == results.csv")


# 12 --------------------------------------------------------------------------------------------
_FORBIDDEN_TRACKED = [
    re.compile(r"^(data|digits|digits_extra|checkpoints|local)/"),
    re.compile(r"\.(pt|pth|ckpt)$"),
    re.compile(r"(^|/)kaggle[^/]*\.json$"),
    re.compile(r"(^|/)credentials[^/]*$"),
    re.compile(r"(^|/)\.env$"),
]
_IMAGE = re.compile(r"\.(jpe?g|png)$", re.I)


def forbidden_tracked(files: list[str]) -> list[str]:
    """Tracked paths that must never be in git (spec section 9.1)."""
    bad = [f for f in files if any(p.search(f) for p in _FORBIDDEN_TRACKED)]
    bad += [f for f in files if _IMAGE.search(f) and not f.startswith(("figures/", "docs/")) and f not in bad]
    return bad


def non_plot_figures(files: list[str], root: Path = ROOT) -> list[str]:
    """Tracked PNGs under figures/ or docs/ that were not written by matplotlib."""
    out = []
    for f in files:
        if f.startswith(("figures/", "docs/")) and _IMAGE.search(f):
            info = Image.open(root / f).info
            if not str(info.get("Software", "")).lower().startswith("matplotlib"):
                out.append(f)
    return out


def git_files(root: Path = ROOT) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln]


def c12_git_clean(files: list[str] | None = None, root: Path = ROOT) -> Result:
    files = git_files(root) if files is None else files
    bad = forbidden_tracked(files) + non_plot_figures(files, root)
    nb = [f for f in files if f.endswith(".ipynb") and notebook_images(root / f) > 0]
    return _ok(not bad and not nb, f"forbidden/non-plot tracked: {bad + nb}" if bad or nb else f"{len(files)} tracked files, none forbidden")


# 13 --------------------------------------------------------------------------------------------
def c13_public(repo: str = "jhlr/hidrometros") -> Result:
    try:
        vis = subprocess.run(
            ["gh", "repo", "view", repo, "--json", "visibility", "-q", ".visibility"], capture_output=True, text=True, check=True
        ).stdout.strip()
        code = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"https://github.com/{repo}"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return Result(False, f"cannot query GitHub: {e}")
    return _ok(vis == "PUBLIC" and code == "200", f"visibility={vis} anonymous HTTP={code}")


CRITERIA: dict[int, tuple[str, Callable[[], Result]]] = {
    1: ("splits.csv sha256 equals the professor's", c01_splits_sha),
    2: ("1243 valid / 1 discarded / 9944 crops, 6 index columns", c02_counts),
    3: ("no photo_name in more than one split", c03_no_leak),
    4: ("corrected-strip counts reported in answer 3.2", c04_orientation_reported),
    5: ("4 models follow the protocol table; seed 42; no flip", c05_protocol),
    6: ("history_*.csv have the 7 columns", c06_history_columns),
    7: ("results.csv 4x9, test-only", c07_results),
    8: ("4 figures exist and are embedded in README", c08_figures),
    9: ("classification_report per model", c09_reports),
    10: ("notebook has no embedded images", c10_notebook),
    11: ("README complete, table == results.csv", c11_readme),
    12: ("git tracks no dataset image/crop/checkpoint/credential", c12_git_clean),
    13: ("repo public and loads logged out", c13_public),
}

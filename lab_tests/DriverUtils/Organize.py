from pathlib import Path
import re
import shutil
from collections import defaultdict


_RUN_DIR_PATTERN = re.compile(r"hls(?P<hls>[^_]+).*lr(?P<lr>[^_]+)", re.IGNORECASE)


def _parse_run_parameters(run_dir_name: str) -> tuple[str, str] | None:
    match = _RUN_DIR_PATTERN.search(run_dir_name)
    if not match:
        return None
    return match.group("hls"), match.group("lr")


def _collect_graph_files(run_dir: Path, valid_exts: set[str]) -> list[Path]:
    graph_files: list[Path] = []
    for graph_type_dir in run_dir.iterdir():
        if not graph_type_dir.is_dir() or len(list(graph_type_dir.iterdir())) > 10:
            continue

        for file_path in graph_type_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in valid_exts:
                graph_files.append(file_path)

    return graph_files


def group_graphs_by_name(output_dir: str) -> None:
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {root}")

    valid_exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}

    parsed_run_dirs: list[tuple[Path, str, str]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue

        run_params = _parse_run_parameters(candidate.name.lower())
        if run_params is None:
            continue

        hls, lr = run_params
        parsed_run_dirs.append((candidate, hls, lr))

    run_dirs_by_hls: dict[str, list[Path]] = defaultdict(list)
    run_dirs_by_lr: dict[str, list[Path]] = defaultdict(list)
    for run_dir, hls, lr in parsed_run_dirs:
        run_dirs_by_hls[hls].append(run_dir)
        run_dirs_by_lr[lr].append(run_dir)

    groupings_root = root / "Groupings"
    groupings_root.mkdir(exist_ok=True)

    unique_hls = set(run_dirs_by_hls.keys())
    unique_lr = set(run_dirs_by_lr.keys())

    create_hls_groups = len(unique_lr) > 1
    create_lr_groups = len(unique_hls) > 1

    if create_hls_groups:
        for hls, run_dirs in run_dirs_by_hls.items():
            _write_group(groupings_root / f"hls{hls}", run_dirs, valid_exts)

    if create_lr_groups:
        for lr, run_dirs in run_dirs_by_lr.items():
            _write_group(groupings_root / f"lr{lr}", run_dirs, valid_exts)


def _write_group(group_dir: Path, run_dirs: list[Path], valid_exts: set[str]) -> None:
    grouped_files: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    group_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in run_dirs:
        for file_path in _collect_graph_files(run_dir, valid_exts):
            grouped_files[file_path.stem].append((run_dir, file_path))

    for graph_name, entries in grouped_files.items():
        if len(entries) <= 1:
            run_dir, file_path = entries[0]
            shutil.copy2(file_path, group_dir / f"{run_dir.name}{file_path.suffix.lower()}")
            continue

        target_dir = group_dir / graph_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for run_dir, file_path in entries:
            shutil.copy2(file_path, target_dir / f"{run_dir.name}{file_path.suffix.lower()}")

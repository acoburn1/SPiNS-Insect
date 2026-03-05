from pathlib import Path
import shutil


def group_graphs_by_name(output_dir: str) -> None:
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {root}")

    valid_exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}

    run_dirs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue

        name = p.name.lower()
        if "hls" in name and "lr" in name:
            run_dirs.append(p)

    for run_dir in run_dirs:
        run_name = run_dir.name

        for graph_type_dir in run_dir.iterdir():
            if not graph_type_dir.is_dir():
                continue

            for file_path in graph_type_dir.iterdir():
                if not file_path.is_file():
                    continue

                ext = file_path.suffix.lower()
                if ext not in valid_exts:
                    continue

                graph_name = file_path.stem
                target_dir = root / graph_name
                target_dir.mkdir(exist_ok=True)

                target_path = target_dir / f"{run_name}{ext}"
                shutil.copy2(file_path, target_path)
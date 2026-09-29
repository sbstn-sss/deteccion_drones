"""Indexar y explorar un dataset YOLO (sin leer imagenes, salvo tamanos)."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_to_label_path(image_path: Path) -> Path:
    """Reemplaza el ULTIMO segmento 'images' de la ruta por 'labels' (convencion Ultralytics)."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"No se encontro un segmento 'images' en la ruta: {image_path}")


def load_data_yaml(yaml_path: str | Path) -> dict:
    """Lee un data.yaml de YOLO y devuelve {root, names, splits}."""
    yaml_path = Path(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = Path(raw["path"])
    if not root.is_absolute():
        root = yaml_path.parent / root

    names = raw["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}

    splits: dict[str, Path | None] = {}
    for split in ("train", "val", "test"):
        value = raw.get(split)
        if value is None:
            splits[split] = None
            continue
        if isinstance(value, list) or str(value).endswith(".txt"):
            raise NotImplementedError(
                f"Split '{split}' es una lista de imagenes o de carpetas ({value!r}); "
                "solo se soporta una carpeta de imagenes por split."
            )
        splits[split] = root / value

    return {"root": root, "names": names, "splits": splits}


def index_split(data: dict, split: str) -> pd.DataFrame:
    """Una fila por imagen del split: split, image_path, label_path, has_label."""
    img_dir = data["splits"].get(split)
    if img_dir is None:
        return pd.DataFrame(columns=["split", "image_path", "label_path", "has_label"])
    if not img_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de imagenes del split '{split}': {img_dir}")

    image_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    rows = []
    for image_path in image_paths:
        label_path = _image_to_label_path(image_path)
        rows.append(
            {
                "split": split,
                "image_path": image_path,
                "label_path": label_path,
                "has_label": label_path.exists(),
            }
        )
    return pd.DataFrame(rows, columns=["split", "image_path", "label_path", "has_label"])


def load_boxes(images: pd.DataFrame, names: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee todos los .txt de `images`. Devuelve (boxes, issues); lineas invalidas van a issues, no se descartan en silencio."""
    box_rows = []
    issue_rows = []

    for _, row in images[images["has_label"]].iterrows():
        with open(row["label_path"], encoding="utf-8") as f:
            lines = f.readlines()

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()

            if len(parts) != 5:
                issue_rows.append(_issue(row["image_path"], line_no, line, "n_cols"))
                continue
            try:
                cls_f, xc, yc, w, h = (float(x) for x in parts)
            except ValueError:
                issue_rows.append(_issue(row["image_path"], line_no, line, "no_numerico"))
                continue

            cls = int(cls_f)
            if cls not in names:
                issue_rows.append(_issue(row["image_path"], line_no, line, "clase_desconocida"))
                continue
            if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
                issue_rows.append(_issue(row["image_path"], line_no, line, "fuera_de_[0,1]"))
                continue
            if w <= 0.0 or h <= 0.0:
                issue_rows.append(_issue(row["image_path"], line_no, line, "w_o_h_cero"))
                continue

            box_rows.append(
                {
                    "image_path": row["image_path"],
                    "split": row["split"],
                    "cls": cls,
                    "name": names[cls],
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                }
            )

    boxes = pd.DataFrame(box_rows, columns=["image_path", "split", "cls", "name", "xc", "yc", "w", "h"])
    issues = pd.DataFrame(issue_rows, columns=["image_path", "line_no", "line", "reason"])
    return boxes, issues


def _issue(image_path: Path, line_no: int, line: str, reason: str) -> dict:
    return {"image_path": image_path, "line_no": line_no, "line": line.rstrip("\n"), "reason": reason}


def add_image_sizes(images: pd.DataFrame, sample: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Agrega columnas width, height leyendo solo el header de cada imagen (PIL)."""
    images = images.copy()
    if sample is not None and sample < len(images):
        idx = random.Random(seed).sample(list(images.index), sample)
    else:
        idx = images.index

    widths = pd.Series(index=images.index, dtype="float64")
    heights = pd.Series(index=images.index, dtype="float64")
    for i in idx:
        with Image.open(images.at[i, "image_path"]) as img:
            widths.at[i], heights.at[i] = img.size

    images["width"] = widths
    images["height"] = heights
    return images


def add_pixel_sizes(boxes: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Agrega w_px, h_px, area_px, size_bucket (criterio COCO) a partir de width/height de `images`.

    Cajas cuya imagen no tiene tamano (fuera del sample de add_image_sizes) se descartan.
    """
    merged = boxes.merge(images[["image_path", "width", "height"]], on="image_path", how="left")
    merged = merged.dropna(subset=["width", "height"])
    merged["w_px"] = merged["w"] * merged["width"]
    merged["h_px"] = merged["h"] * merged["height"]
    merged["area_px"] = merged["w_px"] * merged["h_px"]

    def bucket(area: float) -> str:
        if area < 32**2:
            return "small"
        if area < 96**2:
            return "medium"
        return "large"

    merged["size_bucket"] = merged["area_px"].apply(bucket)
    return merged


def summary(images: pd.DataFrame, boxes: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """Una fila por split: n_images, n_sin_label, n_boxes, boxes_por_img_media, boxes_por_img_max, n_issues."""
    boxes_per_image = boxes.groupby("image_path").size()

    issues_with_split = issues.merge(images[["image_path", "split"]], on="image_path", how="left")

    rows = []
    for split, group in images.groupby("split"):
        counts = boxes_per_image.reindex(group["image_path"], fill_value=0)
        rows.append(
            {
                "split": split,
                "n_images": len(group),
                "n_sin_label": int((~group["has_label"]).sum()),
                "n_boxes": int(counts.sum()),
                "boxes_por_img_media": float(counts.mean()) if len(counts) else 0.0,
                "boxes_por_img_max": int(counts.max()) if len(counts) else 0,
                "n_issues": int((issues_with_split["split"] == split).sum()),
            }
        )
    return pd.DataFrame(rows)


def class_counts(boxes: pd.DataFrame) -> pd.DataFrame:
    """Filas: clase; columnas: splits (n_boxes); agrega columna con % del total de train."""
    table = pd.crosstab(boxes["name"], boxes["split"])
    if "train" in table.columns and table["train"].sum() > 0:
        table["pct_train"] = (table["train"] / table["train"].sum() * 100).round(2)
    else:
        table["pct_train"] = 0.0
    return table


def remap_dataset(yaml_path: str | Path, mapping: dict[str, str | None], out_dir: str | Path) -> Path:
    """Crea una copia del dataset con clases fusionadas, renombradas o eliminadas, por NOMBRE de clase.

    mapping: {"pedestrian": "person", "people": "person"} fusiona; {"DontCare": None} elimina esa clase.
    Las clases que no aparecen en mapping quedan igual. Los indices nuevos siguen el orden de las clases originales.
    Solo se reescriben las labels; las imagenes son hardlinks (no ocupan espacio extra). Nada de symlinks de
    carpeta: Ultralytics resuelve la ruta real del split y terminaria leyendo las labels originales.
    Idempotente (marcador out_dir/.done): si cambia mapping, usar otro out_dir.
    Devuelve out_dir/<nombre de out_dir>.yaml (ese nombre aparece en el nombre de los runs).
    """
    out_dir = Path(out_dir)
    out_yaml = out_dir / f"{out_dir.name}.yaml"
    if (out_dir / ".done").exists():
        return out_yaml

    src = load_data_yaml(yaml_path)
    new_names: list[str] = []
    old_to_new: dict[int, int | None] = {}
    for old_idx, name in sorted(src["names"].items()):
        target = mapping.get(name, name)
        if target is None:
            old_to_new[old_idx] = None
            continue
        if target not in new_names:
            new_names.append(target)
        old_to_new[old_idx] = new_names.index(target)

    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    for split, img_dir in src["splits"].items():
        if img_dir is None:
            continue
        for row in index_split(src, split).itertuples():
            dst_img = out_dir / row.image_path.relative_to(src["root"])
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                try:
                    os.link(row.image_path, dst_img)
                except OSError:  # otro disco o sistema sin hardlinks: copia
                    shutil.copy2(row.image_path, dst_img)
            if row.has_label:
                dst_lbl = out_dir / row.label_path.relative_to(src["root"])
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.write_text(_remap_label_text(row.label_path.read_text(encoding="utf-8"), old_to_new),
                                   encoding="utf-8")
        raw[split] = str(img_dir.relative_to(src["root"]))

    raw["path"] = str(out_dir)
    raw["names"] = dict(enumerate(new_names))
    raw.pop("nc", None)  # Ultralytics lo deduce de names; uno viejo quedaria inconsistente
    out_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out_dir / ".done").touch()
    print(f"Dataset remapeado en {out_dir}: {dict(enumerate(new_names))}")
    return out_yaml


def _remap_label_text(text: str, old_to_new: dict[int, int | None]) -> str:
    out = []
    for line in text.splitlines():
        parts = line.split()
        try:
            new = old_to_new.get(int(float(parts[0])), -1) if parts else -1
        except ValueError:
            new = -1
        if new is None:
            continue  # clase eliminada
        out.append(line if new == -1 else " ".join([str(new), *parts[1:]]))  # -1: linea rara, se deja tal cual
    return "\n".join(out) + ("\n" if out else "")

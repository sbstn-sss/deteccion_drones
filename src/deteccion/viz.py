"""Graficos de exploracion y evaluacion (matplotlib + cv2). No importa ultralytics a nivel de modulo."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

_PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 215, 0),
    (218, 112, 214), (0, 206, 209), (255, 140, 0), (154, 205, 50),
    (219, 112, 147), (100, 149, 237),
]


def _color_for_class(cls: int) -> tuple[int, int, int]:
    return _PALETTE[cls % len(_PALETTE)]


def draw_boxes(
    img_bgr: np.ndarray,
    boxes: pd.DataFrame | np.ndarray,
    names: dict,
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Dibuja cajas normalizadas (xc, yc, w, h) + nombre de clase sobre una copia de img_bgr."""
    img = img_bgr.copy()
    h_img, w_img = img.shape[:2]

    if isinstance(boxes, pd.DataFrame):
        rows = boxes[["cls", "xc", "yc", "w", "h"]].itertuples(index=False)
    else:
        rows = np.asarray(boxes)

    for cls, xc, yc, w, h in rows:
        cls = int(cls)
        x1 = int((xc - w / 2) * w_img)
        y1 = int((yc - h / 2) * h_img)
        x2 = int((xc + w / 2) * w_img)
        y2 = int((yc + h / 2) * h_img)
        box_color = color if color is not None else _color_for_class(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        label = names.get(cls, str(cls))
        cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
    return img


def show_samples(images: pd.DataFrame, boxes: pd.DataFrame, names: dict, n: int = 6, seed: int = 0, cols: int = 3) -> Figure:
    """Muestra n imagenes al azar (seed fija) con sus cajas de ground truth."""
    n = min(n, len(images))
    sample = images.sample(n=n, random_state=seed)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img_bgr = cv2.imread(str(row["image_path"]))
        img_boxes = boxes[boxes["image_path"] == row["image_path"]]
        drawn = draw_boxes(img_bgr, img_boxes, names)
        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{row['image_path'].name}\n{len(img_boxes)} objetos", fontsize=10)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_class_counts(boxes: pd.DataFrame, names: dict) -> Figure:
    """Barras de n_boxes por clase, agrupadas por split."""
    table = pd.crosstab(boxes["name"], boxes["split"])
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(table)), 5))
    table.plot(kind="bar", ax=ax)
    ax.set_xlabel("clase")
    ax.set_ylabel("n_boxes")
    ax.set_title("Cajas por clase y split")
    fig.tight_layout()
    return fig


def plot_box_sizes(boxes_px: pd.DataFrame) -> Figure:
    """Hist 2D w_px vs h_px (log) + barras small/medium/large (criterio COCO)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist2d(boxes_px["w_px"], boxes_px["h_px"], bins=50, norm="log")
    ax1.set_xlabel("ancho (px)")
    ax1.set_ylabel("alto (px)")
    ax1.set_title("Distribucion de tamano de cajas")

    order = ["small", "medium", "large"]
    counts = boxes_px["size_bucket"].value_counts().reindex(order, fill_value=0)
    ax2.bar(order, counts.values)
    ax2.set_ylabel("n_boxes")
    ax2.set_title("Cajas por tamano (COCO)")

    fig.tight_layout()
    return fig


def plot_boxes_per_image(boxes: pd.DataFrame, images: pd.DataFrame) -> Figure:
    """Histograma de numero de cajas por imagen (incluye imagenes sin cajas)."""
    counts = boxes.groupby("image_path").size().reindex(images["image_path"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts.values, bins=min(30, max(counts.max(), 1)))
    ax.set_xlabel("cajas por imagen")
    ax.set_ylabel("n_imagenes")
    ax.set_title("Cajas por imagen")
    fig.tight_layout()
    return fig


def plot_image_sizes(images: pd.DataFrame) -> Figure:
    """Conteo de resoluciones (width x height) distintas."""
    sizes = images.apply(lambda r: f"{int(r['width'])}x{int(r['height'])}", axis=1)
    counts = sizes.value_counts()
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(counts)), 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("resolucion")
    ax.set_ylabel("n_imagenes")
    ax.set_title("Resoluciones de imagen")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def show_run_plots(run_dir: str | Path) -> None:
    """Muestra los png que Ultralytics ya genera en el run (no re-grafica results.csv)."""
    run_dir = Path(run_dir)
    candidates = [run_dir / "results.png", run_dir / "confusion_matrix_normalized.png"]
    candidates += sorted(run_dir.glob("*PR_curve.png"))

    found = [p for p in candidates if p.exists()]
    if not found:
        print(f"No se encontraron plots en {run_dir}")
        return

    fig, axes = plt.subplots(1, len(found), figsize=(6 * len(found), 6))
    axes = np.atleast_1d(axes)
    for ax, path in zip(axes, found):
        ax.imshow(plt.imread(path))
        ax.set_title(path.name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def show_pred_vs_gt(
    model,
    images: pd.DataFrame,
    boxes: pd.DataFrame,
    names: dict,
    n: int = 3,
    seed: int = 0,
    conf: float = 0.25,
    imgsz: int = 640,
) -> Figure:
    """2 filas x n: fila 1 prediccion del modelo, fila 2 ground truth. Titulos con conteos."""
    n = min(n, len(images))
    sample = images.sample(n=n, random_state=seed)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 12))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.reshape(2, n)

    for col, (_, row) in enumerate(sample.iterrows()):
        image_path = row["image_path"]
        results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False)
        pred_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(pred_img)
        axes[0, col].set_title(f"prediccion: {image_path.name}\n{len(results[0].boxes)} objetos", fontsize=10)
        axes[0, col].axis("off")

        img_bgr = cv2.imread(str(image_path))
        img_boxes = boxes[boxes["image_path"] == image_path]
        gt_img = cv2.cvtColor(draw_boxes(img_bgr, img_boxes, names), cv2.COLOR_BGR2RGB)
        axes[1, col].imshow(gt_img)
        axes[1, col].set_title(f"ground truth\n{len(img_boxes)} objetos", fontsize=10)
        axes[1, col].axis("off")

    fig.tight_layout()
    return fig

"""Evaluacion de contrato: el modelo como detector de PERSONAS sobre imagenes completas.

A diferencia de exp.val() (mAP por clase, sobre el dataset de entrenamiento), aqui se mide lo que
usa el sistema: cualquier clase de persona del modelo cuenta como "person", la inferencia es igual
a la de despliegue (frame completo o por recortes) y se reporta P / R / F1 a umbrales de operacion.
Sirve para datasets externos con carpetas de imagenes y labels separadas (ej. Unicamp-UAV).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

PERSON_NAMES = ("person", "pedestrian", "people")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def person_class_ids(names: dict, person_names=PERSON_NAMES) -> list[int]:
    """Indices de las clases del modelo que cuentan como persona."""
    ids = [int(i) for i, n in names.items() if n.lower() in person_names]
    if not ids:
        raise ValueError(f"El modelo no tiene clases de persona {person_names}: {names}")
    return ids


def tiles(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Recortes (x, y, w, h) de tamano uniforme que cubren la imagen; el ultimo de cada eje se pega al borde."""
    stride = max(1, round(tile * (1 - overlap)))

    def starts(dim: int) -> list[int]:
        if dim <= tile:
            return [0]
        s = list(range(0, dim - tile + 1, stride))
        if s[-1] != dim - tile:
            s.append(dim - tile)
        return s

    tw, th = min(tile, width), min(tile, height)
    return [(x, y, tw, th) for y in starts(height) for x in starts(width)]


def predict_persons(model, img_bgr: np.ndarray, class_ids: list[int], conf: float = 0.1, imgsz: int = 1024,
                    tile: int | None = None, overlap: float = 0.2, nms_iou: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Cajas xyxy (px de la imagen completa) y confianzas de personas.

    tile=None: una inferencia sobre el frame completo. tile=1280: recortes en un solo batch.
    Siempre termina con NMS sin distinguir clase: une recortes solapados y clases de persona duplicadas
    (ej. pedestrian y people sobre la misma persona).
    """
    h, w = img_bgr.shape[:2]
    crops = [(0, 0, w, h)] if tile is None else tiles(w, h, tile, overlap)
    results = model.predict(source=[img_bgr[y:y + ch, x:x + cw] for x, y, cw, ch in crops],
                            imgsz=imgsz, conf=conf, classes=class_ids, verbose=False)
    boxes, confs = [], []
    for (x, y, _, _), r in zip(crops, results):
        b = _np(r.boxes.xyxy).reshape(-1, 4)
        boxes.append(b + [x, y, x, y])
        confs.append(_np(r.boxes.conf).reshape(-1))
    boxes, confs = np.concatenate(boxes), np.concatenate(confs)
    if len(boxes) == 0:
        return boxes, confs
    keep = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in boxes],
                            confs.tolist(), conf, nms_iou)
    keep = np.array(keep, dtype=int).reshape(-1)
    return boxes[keep], confs[keep]


def _np(t) -> np.ndarray:
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t, dtype=float)


def read_gt(label_path: Path, width: int, height: int, classes: list[int] | None = None) -> np.ndarray:
    """Labels YOLO normalizados -> xyxy en px. classes=None: todas las lineas son personas."""
    if not label_path.exists():
        return np.empty((0, 4))
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5 or (classes is not None and int(float(p[0])) not in classes):
            continue
        xc, yc, bw, bh = (float(v) for v in p[1:5])
        rows.append([(xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height])
    return np.array(rows, dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(pred: np.ndarray, confs: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> tuple[int, int, int]:
    """(TP, FP, FN). Greedy por confianza: cada prediccion toma la persona libre con mayor IoU >= iou_thr."""
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt)
    iou = iou_matrix(pred[np.argsort(-confs)], gt)
    free = np.ones(len(gt), dtype=bool)
    tp = 0
    for row in iou:
        cand = np.where(free, row, -1.0)
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            free[j] = False
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def pair_images(images_dir: str | Path, labels_dir: str | Path) -> pd.DataFrame:
    """Una fila por imagen: image_path, label_path (mismo nombre, .txt), has_label."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    df = pd.DataFrame({"image_path": imgs, "label_path": [labels_dir / f"{p.stem}.txt" for p in imgs]})
    df["has_label"] = df["label_path"].map(Path.exists)
    return df


def evaluate_persons(model, images_dir, labels_dir, conf_thresholds=(0.25, 0.35, 0.5), imgsz: int = 1024,
                     tile: int | None = None, overlap: float = 0.2, iou_thr: float = 0.5,
                     gt_classes: list[int] | None = None, limit: int | None = None, seed: int = 0) -> pd.DataFrame:
    """P / R / F1 / TP / FP / FN de personas por umbral de confianza, sobre imagenes completas.

    Infiere una sola vez con el umbral mas bajo y filtra para cada umbral. limit: muestra al azar (prueba rapida).
    """
    pairs = pair_images(images_dir, labels_dir)
    if limit:
        pairs = pairs.sample(n=min(limit, len(pairs)), random_state=seed)
    ids = person_class_ids(model.names)
    thresholds = sorted(conf_thresholds)
    counts = {t: [0, 0, 0] for t in thresholds}
    t0 = time.time()
    for i, row in enumerate(pairs.itertuples(), 1):
        img = cv2.imread(str(row.image_path))
        if img is None:
            print(f"No se pudo leer {row.image_path}")
            continue
        gt = read_gt(row.label_path, img.shape[1], img.shape[0], gt_classes)
        boxes, confs = predict_persons(model, img, ids, thresholds[0], imgsz, tile, overlap)
        for t in thresholds:
            keep = confs >= t
            for k, v in enumerate(match(boxes[keep], confs[keep], gt, iou_thr)):
                counts[t][k] += v
        if i % 100 == 0 or i == len(pairs):
            print(f"{i}/{len(pairs)} imagenes, {(time.time() - t0) / i:.2f} s/img")

    rows = []
    for t, (tp, fp, fn) in counts.items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"conf": t, "P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
                     "TP": tp, "FP": fp, "FN": fn, "n_gt": tp + fn, "n_imgs": len(pairs)})
    return pd.DataFrame(rows).set_index("conf")


def show_persons(model, images_dir, labels_dir, n: int = 3, seed: int = 0, conf: float = 0.35, imgsz: int = 1024,
                 tile: int | None = None, overlap: float = 0.2, gt_classes: list[int] | None = None) -> Figure:
    """n imagenes al azar: ground truth en azul, predicciones en verde. Titulo con TP / FP / FN."""
    pairs = pair_images(images_dir, labels_dir)
    sample = pairs.iloc[random.Random(seed).sample(range(len(pairs)), min(n, len(pairs)))]
    ids = person_class_ids(model.names)
    fig, axes = plt.subplots(len(sample), 1, figsize=(16, 9 * len(sample)))
    for ax, row in zip(np.atleast_1d(axes), sample.itertuples()):
        img = cv2.imread(str(row.image_path))
        gt = read_gt(row.label_path, img.shape[1], img.shape[0], gt_classes)
        boxes, confs = predict_persons(model, img, ids, conf, imgsz, tile, overlap)
        tp, fp, fn = match(boxes, confs, gt)
        th = max(2, img.shape[1] // 800)
        for x1, y1, x2, y2 in gt.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 0), th)
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), th)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{row.image_path.name}  |  TP {tp}  FP {fp}  FN {fn}  (azul = real, verde = modelo)")
        ax.axis("off")
    fig.tight_layout()
    plt.close(fig)
    return fig

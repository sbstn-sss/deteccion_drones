"""Entorno Colab / Drive / datasets: montaje, descarga y descompresion idempotentes."""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

import yaml

IN_COLAB = "google.colab" in sys.modules
DRIVE_ROOT = Path("/content/drive")
DATA_ROOT = Path("/content/datasets")


def mount_drive() -> None:
    """Monta Google Drive en Colab (no hace nada fuera de Colab)."""
    if not IN_COLAB:
        return
    if (DRIVE_ROOT / "MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(str(DRIVE_ROOT))
    except Exception as exc:
        raise RuntimeError(
            "No se pudo montar Drive desde la extension de Colab para VS Code. "
            "Prueba abrir el notebook en Colab web (desde GitHub)."
        ) from exc


def drive_path(p: str | Path) -> Path:
    """Resuelve una ruta relativa a DRIVE_ROOT (ej 'MyDrive/x'); absoluta -> tal cual."""
    p = Path(p)
    return p if p.is_absolute() else DRIVE_ROOT / p


def _is_done(dest: Path) -> bool:
    return (dest / ".done").exists()


def _mark_done(dest: Path) -> None:
    (dest / ".done").touch()


def unzip_once(zip_path: str | Path, dest: str | Path) -> Path:
    """Descomprime zip_path en dest si dest/.done no existe (una extraccion cortada a la mitad se re-hace)."""
    zip_path = Path(zip_path)
    dest = Path(dest)
    if _is_done(dest):
        return dest
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontro el zip: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    _mark_done(dest)
    print(f"Descomprimido {zip_path} -> {dest} en {time.time() - start:.1f}s")
    return dest


def kaggle_download_once(slug: str, dest: str | Path) -> Path:
    """Descarga y descomprime un dataset de Kaggle en dest si dest/.done no existe."""
    dest = Path(dest)
    if _is_done(dest):
        return dest

    if IN_COLAB:
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        except Exception as exc:
            raise RuntimeError(
                "No se pudieron leer las credenciales de Kaggle desde los Secrets de Colab "
                "(KAGGLE_USERNAME, KAGGLE_KEY)."
            ) from exc

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError(
            "Falta el paquete 'kaggle'. Instalar con: pip install 'deteccion[kaggle]'"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    _mark_done(dest)
    return dest


def fix_data_yaml(yaml_path: str | Path, root: str | Path | None = None) -> Path:
    """Corrige el campo 'path' de un data.yaml de YOLO y escribe '<stem>_local.yaml' (no toca el original)."""
    yaml_path = Path(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["path"] = str(root) if root is not None else str(yaml_path.parent)

    out_path = yaml_path.with_name(f"{yaml_path.stem}_local.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return out_path

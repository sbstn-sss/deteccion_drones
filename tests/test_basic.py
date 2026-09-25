"""Asserts planos. Ejecutar con: python tests/test_basic.py (sin GPU, sin dataset, sin ultralytics)."""

import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display; debe fijarse antes de que deteccion.viz importe pyplot

import pandas as pd
import yaml
from PIL import Image

from deteccion import data, env, viz


def test_load_boxes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        label_path = tmp / "labels" / "train" / "img1.txt"
        label_path.parent.mkdir(parents=True)
        label_path.write_text(
            "\n".join(
                [
                    "0 0.5 0.5 0.2 0.2",  # valida
                    "1 0.5 0.5 0.2",  # n_cols
                    "0 0.5 1.3 0.2 0.2",  # fuera_de_[0,1]
                    "99 0.5 0.5 0.2 0.2",  # clase_desconocida
                    "abc def ghi jkl mno",  # no_numerico
                ]
            )
        )

        images = pd.DataFrame(
            [{"image_path": tmp / "images" / "train" / "img1.jpg", "split": "train",
              "label_path": label_path, "has_label": True}]
        )
        names = {0: "pedestrian", 1: "car"}

        boxes, issues = data.load_boxes(images, names)

        assert len(boxes) == 1, f"esperaba 1 box valida, obtuve {len(boxes)}"
        assert boxes.iloc[0]["cls"] == 0
        assert len(issues) == 4, f"esperaba 4 issues, obtuve {len(issues)}"
        reasons = set(issues["reason"])
        assert reasons == {"n_cols", "fuera_de_[0,1]", "clase_desconocida", "no_numerico"}, reasons

    print("test_load_boxes OK")


def test_index_split():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # "images" aparece tambien en un directorio padre, no solo en el ultimo segmento.
        img_dir = tmp / "images_root" / "images" / "train"
        img_dir.mkdir(parents=True)
        img_path = img_dir / "img1.jpg"
        img_path.write_bytes(b"")

        data_dict = {
            "root": tmp,
            "names": {0: "pedestrian"},
            "splits": {"train": img_dir, "val": None, "test": None},
        }
        images = data.index_split(data_dict, "train")

        assert len(images) == 1
        expected_label = tmp / "images_root" / "labels" / "train" / "img1.txt"
        assert images.iloc[0]["label_path"] == expected_label, images.iloc[0]["label_path"]
        assert not images.iloc[0]["has_label"]

    print("test_index_split OK")


def test_fix_data_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        yaml_path = tmp / "visdrone.yaml"
        original = {"path": "/content/dataset/VisDrone_Dataset", "train": "images/train", "names": ["pedestrian"]}
        yaml_path.write_text(yaml.safe_dump(original))

        out_path = env.fix_data_yaml(yaml_path, root=tmp / "local_root")

        assert out_path == tmp / "visdrone_local.yaml"
        assert out_path.exists()

        fixed = yaml.safe_load(out_path.read_text())
        assert fixed["path"] == str(tmp / "local_root")

        untouched = yaml.safe_load(yaml_path.read_text())
        assert untouched["path"] == "/content/dataset/VisDrone_Dataset", "el original no debe modificarse"

    print("test_fix_data_yaml OK")


def test_add_pixel_sizes_sample():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_dir = tmp / "images"
        img_dir.mkdir()

        rows = []
        for i, (w, h) in enumerate([(100, 50), (60, 60), (40, 20)]):
            p = img_dir / f"img{i}.jpg"
            Image.new("RGB", (w, h), "red").save(p)
            rows.append({"image_path": p, "split": "train", "label_path": p, "has_label": True})
        images = pd.DataFrame(rows)

        images_sz = data.add_image_sizes(images, sample=1, seed=0)
        sampled = images_sz.dropna(subset=["width", "height"])
        assert len(sampled) == 1, f"esperaba 1 imagen con tamano, obtuve {len(sampled)}"

        boxes = pd.DataFrame(
            [
                {"image_path": r["image_path"], "split": "train", "cls": 0, "name": "x",
                 "xc": 0.5, "yc": 0.5, "w": 0.1, "h": 0.1}
                for _, r in images.iterrows()
            ]
        )

        boxes_px = data.add_pixel_sizes(boxes, images_sz)
        assert len(boxes_px) == 1, f"esperaba solo la caja de la imagen muestreada, obtuve {len(boxes_px)}"
        assert boxes_px.iloc[0]["image_path"] == sampled.iloc[0]["image_path"]

        sampled_row = sampled.iloc[0]
        expected_area = (0.1 * sampled_row["width"]) * (0.1 * sampled_row["height"])
        expected_bucket = "small" if expected_area < 32**2 else ("medium" if expected_area < 96**2 else "large")
        assert boxes_px.iloc[0]["size_bucket"] == expected_bucket, boxes_px.iloc[0]["size_bucket"]

        fig = viz.plot_image_sizes(images_sz)  # no debe lanzar pese a las 2 filas sin width/height
        assert fig is not None

    print("test_add_pixel_sizes_sample OK")


def test_unzip_once():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.txt", "hola")
        dest = tmp / "dest"

        env.unzip_once(zip_path, dest)
        assert (dest / "a.txt").exists()
        assert (dest / ".done").exists()

        (dest / "a.txt").unlink()
        env.unzip_once(zip_path, dest)
        assert not (dest / "a.txt").exists(), "no deberia haber re-extraido (.done presente)"

        (dest / ".done").unlink()
        env.unzip_once(zip_path, dest)
        assert (dest / "a.txt").exists(), "deberia re-extraer si faltaba .done, aunque dest tuviera archivos"

    print("test_unzip_once OK")


if __name__ == "__main__":
    test_load_boxes()
    test_index_split()
    test_fix_data_yaml()
    test_add_pixel_sizes_sample()
    test_unzip_once()
    print("Todos los tests pasaron.")

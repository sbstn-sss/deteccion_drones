"""Asserts planos. Ejecutar con: python tests/test_basic.py (sin GPU, sin dataset, sin ultralytics)."""

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from deteccion import data, env


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


if __name__ == "__main__":
    test_load_boxes()
    test_index_split()
    test_fix_data_yaml()
    print("Todos los tests pasaron.")

"""Un experimento = un run de Ultralytics en Drive: entrenar, retomar, cargar, validar y listar runs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from deteccion import data as data_mod
from deteccion import env, viz

SORT_KEYS = {"date", "mAP50", "mAP50-95"}


def _yolo(weights: str | Path):
    # ultralytics se importa aqui para que el resto del paquete y los tests corran sin torch.
    from ultralytics import YOLO

    return YOLO(str(weights))


@dataclass
class ExperimentConfig:
    data_yaml: str
    out_dir: str
    model: str = "yolo11s.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    seed: int = 0
    tag: str = ""
    train_args: dict = field(default_factory=dict)

    def run_name(self, stamp: str) -> str:
        """'{dataset}_{modelo}_{imgsz}[_{tag}]_{stamp}', ej 'visdrone_yolo11s_1024_cosLR_20260925-1530'."""
        yaml_path = Path(self.data_yaml)
        dataset = yaml_path.stem.removesuffix("_local")
        if dataset in ("data", "dataset"):  # nombre generico (ej HIT-UAV): usar la carpeta, "hit-uav"
            dataset = yaml_path.parent.name
        parts = [dataset, Path(self.model).stem, str(self.imgsz)]
        if self.tag:
            parts.append(self.tag)
        parts.append(stamp)
        return "_".join(parts)


class Experiment:
    def __init__(self, cfg: ExperimentConfig, run_name: str | None = None):
        self.cfg = cfg
        self.run_name = run_name or cfg.run_name(datetime.now().strftime("%Y%m%d-%H%M"))
        self.run_dir = env.drive_path(cfg.out_dir) / self.run_name
        self._model = None

    @property
    def best(self) -> Path:
        return self.run_dir / "weights" / "best.pt"

    @property
    def last(self) -> Path:
        return self.run_dir / "weights" / "last.pt"

    @property
    def model(self):
        """YOLO(best.pt), cargado una sola vez."""
        if self._model is None:
            if not self.best.exists():
                raise FileNotFoundError(f"Este run no tiene best.pt: {self.best}")
            self._model = _yolo(self.best)
        return self._model

    def train(self):
        """Entrena desde cfg.model y guarda el run en run_dir. Devuelve los results de Ultralytics."""
        if "augment" in self.cfg.train_args:
            raise ValueError(
                "No pasar 'augment' a train: es TTA de prediccion, no aumento de datos. "
                "El aumento se controla con mosaic, fliplr, hsv_h, etc."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "experiment.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")

        results = _yolo(self.cfg.model).train(
            data=self.cfg.data_yaml,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=self.cfg.batch,
            seed=self.cfg.seed,
            project=str(self.run_dir.parent),
            name=self.run_name,
            exist_ok=True,
            **self.cfg.train_args,
        )
        self._model = None
        return results

    def resume(self):
        """Retoma un entrenamiento cortado (ej. desconexion de Colab) desde last.pt."""
        if not self.last.exists():
            raise FileNotFoundError(f"No hay last.pt para retomar: {self.last}")
        results = _yolo(self.last).train(resume=True)
        self._model = None
        return results

    @classmethod
    def load(cls, run_dir: str | Path, data_yaml: str | None = None) -> "Experiment":
        """Reconstruye el experimento desde experiment.json o, en runs viejos, desde args.yaml."""
        run_dir = env.drive_path(run_dir)
        exp_json = run_dir / "experiment.json"
        args_yaml = run_dir / "args.yaml"

        if exp_json.exists():
            cfg = ExperimentConfig(**json.loads(exp_json.read_text(encoding="utf-8")))
        elif args_yaml.exists():
            args = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
            cfg = ExperimentConfig(
                data_yaml=args["data"], out_dir="", model=args["model"],
                epochs=args["epochs"], imgsz=args["imgsz"], batch=args["batch"],
            )
        else:
            raise FileNotFoundError(f"{run_dir} no tiene experiment.json ni args.yaml: no parece un run")

        # La carpeta real manda sobre la guardada (las carpetas de Drive se pueden mover).
        cfg.out_dir = str(run_dir.parent)
        if data_yaml is not None:
            cfg.data_yaml = str(data_yaml)
        return cls(cfg, run_name=run_dir.name)

    def val(self, split: str = "val", conf: float = 0.001) -> pd.DataFrame:
        """Valida best.pt en un split. Devuelve fila 'all' + una por clase con P, R, mAP50, mAP50-95."""
        if data_mod.load_data_yaml(self.cfg.data_yaml)["splits"].get(split) is None:
            raise ValueError(f"El yaml {self.cfg.data_yaml} no tiene split '{split}'")

        metrics = self.model.val(
            data=self.cfg.data_yaml, split=split, imgsz=self.cfg.imgsz, conf=conf,
            project=str(self.run_dir), name=f"eval_{split}", exist_ok=True,
            classes=self.cfg.train_args.get("classes"),  # mismas clases que en train (ej. sin DontCare en IR)
        )
        box = metrics.box
        rows = [{"clase": "all", "P": box.mp, "R": box.mr, "mAP50": box.map50, "mAP50-95": box.map}]
        for i, c in enumerate(metrics.ap_class_index):
            p, r, ap50, ap = box.class_result(i)
            rows.append({"clase": metrics.names[int(c)], "P": p, "R": r, "mAP50": ap50, "mAP50-95": ap})
        return pd.DataFrame(rows).set_index("clase")

    def show_pred_vs_gt(self, split: str = "test", n: int = 3, seed: int = 0, conf: float = 0.25):
        """n imagenes al azar del split: prediccion de best.pt arriba, ground truth abajo."""
        d = data_mod.load_data_yaml(self.cfg.data_yaml)
        if d["splits"].get(split) is None:
            raise ValueError(f"El yaml {self.cfg.data_yaml} no tiene split '{split}'")
        images = data_mod.index_split(d, split)
        sample = images.sample(n=min(n, len(images)), random_state=seed)  # solo se leen los labels de la muestra
        boxes, _ = data_mod.load_boxes(sample, d["names"])
        return viz.show_pred_vs_gt(self.model, sample, boxes, d["names"], n=n, seed=seed, conf=conf, imgsz=self.cfg.imgsz)

    def _video_out(self, video: str | Path, kind: str) -> tuple[Path, Path, str]:
        video = env.drive_path(video)
        if not video.exists():
            raise FileNotFoundError(f"No se encontro el video: {video}")
        return video, self.run_dir / "videos", f"{video.stem}_{kind}"

    def predict_video(self, video: str | Path, conf: float = 0.25, **kw) -> Path:
        """Detecta en cada frame y guarda el video anotado en run_dir/videos/<video>_predict."""
        video, project, name = self._video_out(video, "predict")
        # stream=True: no acumula los resultados de todos los frames en RAM.
        for _ in self.model.predict(source=str(video), imgsz=self.cfg.imgsz, conf=conf, save=True, stream=True,
                                    project=str(project), name=name, exist_ok=True, verbose=False, **kw):
            pass
        return project / name

    def track(self, video: str | Path, tracker: str = "botsort.yaml", conf: float = 0.15, iou: float = 0.5,
              **kw) -> Path:
        """Tracking en el video; guarda el video con IDs en run_dir/videos/<video>_track."""
        video, project, name = self._video_out(video, "track")
        results = self.model.track(source=str(video), imgsz=self.cfg.imgsz, conf=conf, iou=iou, tracker=tracker,
                                   save=True, stream=True, persist=True, project=str(project), name=name,
                                   exist_ok=True, verbose=False, **kw)
        for i, r in enumerate(results):
            if i % 30 == 0:
                n_ids = 0 if r.boxes.id is None else len(r.boxes.id)
                print(f"frame {i}: {n_ids} objetos con id")
        return project / name


def list_runs(out_dir: str | Path, sort: str = "date") -> pd.DataFrame:
    """Una fila por run (subcarpeta con args.yaml) de out_dir, con sus mejores metricas."""
    if sort not in SORT_KEYS:
        raise ValueError(f"sort debe ser uno de {sorted(SORT_KEYS)}, no {sort!r}")
    out_dir = env.drive_path(out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de runs: {out_dir}")

    rows = []
    for run in sorted(p for p in out_dir.iterdir() if (p / "args.yaml").exists()):
        args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        results_csv = run / "results.csv"
        epochs_done, map50, map5095 = 0, math.nan, math.nan
        if results_csv.exists():
            res = pd.read_csv(results_csv)
            res.columns = res.columns.str.strip()  # versiones viejas de Ultralytics traen espacios
            epochs_done = len(res)
            map50 = res["metrics/mAP50(B)"].max() if "metrics/mAP50(B)" in res else math.nan
            map5095 = res["metrics/mAP50-95(B)"].max() if "metrics/mAP50-95(B)" in res else math.nan
        stamp_file = results_csv if results_csv.exists() else run / "args.yaml"
        rows.append({
            "name": run.name, "path": run, "model": args.get("model"), "data": args.get("data"),
            "imgsz": args.get("imgsz"), "epochs": args.get("epochs"), "epochs_done": epochs_done,
            "mAP50": map50, "mAP50-95": map5095, "has_best": (run / "weights" / "best.pt").exists(),
            "date": datetime.fromtimestamp(stamp_file.stat().st_mtime),
        })

    cols = ["name", "path", "model", "data", "imgsz", "epochs", "epochs_done",
            "mAP50", "mAP50-95", "has_best", "date"]
    runs = pd.DataFrame(rows, columns=cols)
    return runs.sort_values(sort, ascending=False, na_position="last").reset_index(drop=True)

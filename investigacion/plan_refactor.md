---
title: "Plan: refactor de notebooks a modulo src reutilizable (RGB + termico)"
fecha: 2026-09-25
estado: F0 hecho, siguiente F1
---

# Plan de refactor

> **Spec implementable (fuente de verdad): `docs/SPEC.md`.** Este archivo queda como notas/razonamiento.

## 0. Diagnostico de `hola.ipynb` y `mot_yolo.ipynb`

| # | Falencia | Donde | Arreglo |
|---|----------|-------|---------|
| 1 | Rutas hardcodeadas repetidas (MODEL_PATH se define 3 veces, una sobreescribe a otra) | celdas 9, 17, 19, 21 | Una celda de parametros arriba; el resto se deriva |
| 2 | Nombre de run `...opt-8` escrito a mano: Ultralytics auto-incrementa `-2, -3...` y tienes que adivinar | celda 17 | Nombre de run generado + `exist_ok`; la clase sabe donde queda `best.pt` |
| 3 | Markdown dice YOLOv8, codigo carga `yolo11s.pt`; `config.yaml` dice `yolov8s` y batch 16, notebook usa batch 8 | celdas 12-15 | El modelo es UN parametro (`"yolov8s.pt"`, `"yolo11s.pt"`...) |
| 4 | Parcheo del `path:` del yaml linea por linea como texto | celda 7 | Leer con `yaml.safe_load`, escribir un yaml nuevo con el path correcto |
| 5 | `!mkdir` + `!unzip` fallan / re-descomprimen si re-corres | celda 4 | Setup idempotente: si ya existe, no hace nada |
| 6 | Dibujo de cajas YOLO copiado 2 veces, sin nombres de clase, y `len(data)==5` descarta lineas en silencio | celdas 11, 17 | Una funcion `draw_boxes()` en `viz.py` |
| 7 | `if 'df' in locals()`: la celda depende de estado oculto | celda 11 | Funciones que reciben lo que necesitan como argumento |
| 8 | `augment=True` en `train()` no hace lo que parece (es TTA de *prediccion*); el aumento en train se controla con `mosaic`, `fliplr`, `hsv_*`, etc. | celda 15 | Quitarlo; exponer los hiperparams de aumento reales |
| 9 | Evaluacion solo visual (3 imagenes al azar, sin semilla). No hay mAP sobre test | celda 17 | `model.val(split="test")` + visual con `seed` |
| 10 | El modelo se carga de nuevo en cada celda | 17, 19, 21 | Se carga una vez en el objeto `Experiment` |
| 11 | `mot_yolo`: `cv2.imread` + `cv2.imwrite` solo para copiar re-comprime el JPG (lento y pierde calidad) | celda 3 | `shutil.copy` y leer el tamano 1 vez por secuencia |
| 12 | `mot_yolo`: no filtra la columna `score` (col 7) = 0, que en VisDrone-MOT marca cajas **ignoradas** | celda 3 | `if score == 0: continue` |
| 13 | `hola.ipynb` pesa 7.8 MB por las imagenes en outputs, y esta en git | repo | `nbstripout` (limpia outputs al commitear) |

## 1. Estructura propuesta

```
deteccion/
├── pyproject.toml             # permite `pip install -e .` -> `from deteccion import ...` en cualquier lado
├── src/deteccion/             # TODO el codigo reutilizable (sin celdas, sin rutas fijas)
│   ├── __init__.py
│   ├── config.py              # @dataclass ExperimentConfig
│   ├── env.py                 # detectar Colab, montar Drive, descomprimir dataset (idempotente)
│   ├── data.py                # indexar dataset YOLO -> DataFrame, estadisticas, validar labels
│   ├── viz.py                 # draw_boxes, show_samples, plot_class_dist, plot_box_sizes, pred_vs_gt
│   ├── experiment.py          # class Experiment: train / val / predict / track
│   └── converters/
│       └── visdrone_mot.py    # lo de mot_yolo.ipynb, arreglado
├── notebooks/                 # solo "llamadas + graficos", nada de logica
│   ├── rgb/                   # VisDrone (DET y MOT)
│   │   ├── 01_explorar.ipynb
│   │   ├── 02_entrenar.ipynb
│   │   └── 03_testing.ipynb
│   └── ir/                    # HIT-UAV
│       ├── 01_explorar.ipynb
│       ├── 02_entrenar.ipynb
│       └── 03_testing.ipynb
├── outdated/                  # hola.ipynb, mot_yolo.ipynb, config.yaml viejos (gitignored)
└── investigacion/
```

Mas adelante (NO ahora, cuando llegue el momento):
- `src/deteccion/thermal/hotspots.py` -> deteccion de hotspots con OpenCV (umbral + contornos + centroide -> direccion)
- `src/deteccion/thermal/preprocess.py` -> frame de camara -> mismo formato que HIT-UAV (8-bit gris, white-hot).
  Depende de lo que entregue la camara: raw 16-bit (normalizar con rango fijo), gris 8-bit (directo),
  o paleta de color (invertir la paleta con LUT, NO cvtColor a gris). Hotspots idealmente sobre el raw 16-bit.
- `converters/<dataset_termico>.py` si el dataset termico no viene en formato YOLO
- `notebooks/10_hotspots_ir.ipynb`

> Por que notebooks **fuera** de `src/`: `src/` es el paquete que se importa. Si el notebook vive
> adentro, termina importandose a si mismo por rutas relativas raras. Convencion estandar:
> `src/` = libreria, `notebooks/` = quien la usa.

## 2. Como se veria un notebook (la API objetivo)

```python
# ── Celda 1: PARAMETROS (lo unico que editas) ─────────────────────────
DATASET_ZIP = "MyDrive/deteccion_drones_v1/archive.zip"
DATA_YAML   = "VisDrone_Dataset/visdrone.yaml"   # relativo a donde se descomprime
OUT_DIR     = "MyDrive/VisDrone_Prototipo"
MODEL       = "yolo11s.pt"        # o "yolov8s.pt", "yolov8n.pt"...
EPOCHS, IMGSZ, BATCH = 50, 1024, 8

# ── Celda 2: entorno ──────────────────────────────────────────────────
%load_ext autoreload
%autoreload 2
from deteccion import env, ExperimentConfig, Experiment
env.setup(repo="https://github.com/sbstn-sss/deteccion_drones.git", dataset_zip=DATASET_ZIP)

# ── Celda 3: experimento ─────────────────────────────────────────────
cfg = ExperimentConfig(data_yaml=DATA_YAML, out_dir=OUT_DIR, model=MODEL,
                       epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                       train_args=dict(cos_lr=True, close_mosaic=15, patience=10))
exp = Experiment(cfg)          # nombre del run auto: visdrone_yolo11s_1024_20260925-1530
exp.train()
exp.val(split="test")          # mAP real
exp.show_pred_vs_gt(n=3, seed=0)
exp.track("MyDrive/deteccion_drones_v1/videos/video1_testing.mp4", tracker="botsort.yaml")
```

Para reusar un run ya entrenado (sin re-entrenar):
```python
exp = Experiment.load("MyDrive/VisDrone_Prototipo/visdrone_yolo11s_1024_20260925-1530")
```

### Separacion de notebooks (cada uno independiente, se corre solo)

Mismos 3 notebooks en `notebooks/rgb/` y `notebooks/ir/`: cambia la celda de parametros
(y cosas propias del dominio, ej. que hacer con `DontCare` en IR). El codigo de `src/` es el mismo.

| Notebook | Hace | NO hace |
|----------|------|---------|
| `01_explorar_dataset` | setup + stats + graficos del dataset | no carga modelos |
| `02_entrenar` | setup + `exp.train()` + curvas de entrenamiento | no testea ni trackea |
| `03_testing` | setup + elegir run + mAP en test + pred vs GT + video/tracking | no entrena |

Cada uno tiene su propia celda de parametros y su propio `env.setup()`: no dependen de
haber corrido otro notebook en la misma sesion.

En `03_testing` el modelo NO se hardcodea:
```python
runs = list_runs(OUT_DIR)      # tabla: nombre, modelo, dataset, imgsz, epocas, mAP50, mAP50-95, fecha
runs                           # la ves y eliges
exp = Experiment.load(runs.iloc[0].path)   # o por nombre, o list_runs(..., sort="mAP50-95")
```
`list_runs` lee `args.yaml` + `results.csv` que Ultralytics ya deja en cada run.

Para el termico: **cambias solo la celda 1** (otro zip, otro yaml, `MODEL="yolov8s.pt"`).

Detalles de diseno:
- `ExperimentConfig` es un `dataclass`: autocompleta en el editor, falla si escribes mal un campo, y se guarda en el run.
- `train_args` es un dict que va directo a `YOLO.train()` -> cualquier hiperparam de Ultralytics sin tocar la clase.
- Ultralytics ya guarda `args.yaml` y `results.csv` en cada run -> eso ES el registro del experimento. No hace falta reinventar tracking (W&B opcional).
- Una sola clase (`Experiment`). Lo demas son funciones sueltas: exploracion y graficos no necesitan estado.

## 3. Buenas practicas de Jupyter (las que aplican aqui)

1. **Logica en `.py`, notebook solo orquesta y muestra.** Si una celda pasa de ~15 lineas, es una funcion.
2. **Parametros solo en la primera celda.** Ninguna ruta literal mas abajo.
3. **"Restart & Run All" debe funcionar.** Si no, hay estado oculto (el `if 'df' in locals()`).
4. **Celdas idempotentes**: correrlas 2 veces no rompe nada (no re-descomprimir, `exist_ok=True`).
5. **`%autoreload 2`**: editas el `.py` y la siguiente llamada ya usa el cambio, sin reiniciar el kernel.
6. **No cargar datos a la fuerza**: el DataFrame guarda *rutas*, no imagenes. Las imagenes se leen solo al graficar una muestra.
7. **Semillas** en todo lo aleatorio (muestras a visualizar, splits).
8. **Un notebook por etapa**, numerados. Explorar != entrenar != evaluar.
9. **Outputs fuera de git** (`nbstripout`). Los graficos importantes se guardan como png en el run.

## 4. Pipeline general para entrenar modelos (vision o cualquiera)

```
1. Datos        -> conseguir, convertir a formato comun (YOLO), NUNCA tocar el test
2. Explorar     -> clases desbalanceadas? cajas diminutas? imagenes sin label? labels rotos?
3. Baseline     -> modelo chico (n/s), pocas epocas, params por defecto. Es tu piso.
4. Experimentar -> cambiar UNA cosa a la vez (imgsz, modelo, aumento). Nombre de run descriptivo.
5. Evaluar      -> metrica en val para decidir, en test solo al final. + inspeccion visual.
6. Analizar err -> donde falla? (clase, tamano, oclusion) -> vuelve a 2 o 4
7. Desplegar    -> tracking en video, exportar (onnx/tensorrt) si va al dron
```

Tus hallazgos de exploracion (ej: VisDrone tiene muchisimos objetos < 32px) son los que justifican
decisiones como `imgsz=1024`. Eso es lo que `data.py` + `viz.py` deben hacer facil de ver.

## 5. Fases de implementacion

| Fase | Que | Listo cuando |
|------|-----|--------------|
| F0 | Limpiar repo: mover notebooks a `legacy/`, `pyproject.toml`, nbstripout, decidir sync a Colab | `pip install -e .` funciona local |
| F1 | `config.py` + `env.py` + `data.py` + `viz.py` -> `01_explorar_dataset.ipynb` con VisDrone | Restart & Run All en Colab sin editar nada salvo celda 1 |
| F2 | `experiment.py` (train/val/load) -> `02_entrenar.ipynb` | Re-entrena yolo11s y encuentra `best.pt` solo |
| F3 | predict/track + pred vs GT -> `03_evaluar_y_trackear.ipynb` | Reproduce los videos que ya tienes |
| F4 | `converters/visdrone_mot.py` con los bugs 11-12 arreglados | MOT convertido y entrenable con el mismo `02` |
| F5 | Dataset termico: solo celda 1 (+ converter si hace falta) | Entrena yolov8s en IR sin tocar `src/` |
| F6 | `thermal/hotspots.py` + `thermal/preprocess.py` | **FUERA DE ALCANCE por ahora**: sin acceso a datos de camaras. Algoritmo y formato se definen despues |

**Alcance actual: F0-F5** = framework de entrenamiento + notebooks ordenados (VisDrone y HIT-UAV).

## 6. Decisiones pendientes

- [x] Sync de codigo: GitHub `sbstn-sss/deteccion_drones`. Notebooks se editan en VS Code con kernel Colab (extension).
      Ojo: el kernel corre en la VM de Colab, no ve tus archivos locales -> editas src/ -> push -> la celda de setup hace `git pull`.
- [x] Dataset termico: **HIT-UAV** (640x512, clases Person, Car, Bicycle, OtherVehicle, DontCare).
      Revisar al explorar si la version descargada ya trae labels YOLO (Kaggle si) -> sin converter.
      Decidir que hacer con `DontCare` (quitarla o dejarla).
      `env.setup` acepta zip en Drive **o** slug de Kaggle (descarga directa a /content).
- [ ] W&B: mantener o basta con `results.csv` de Ultralytics
- [ ] Formato del stream termico que llega al outpost (raw 16-bit / gris 8-bit / paleta, white/black-hot,
      resolucion, fps, protocolo). Pendiente de conversar. Bloquea solo `thermal/preprocess.py`, no F0-F5.
      Preguntas a llevar: ¿la camara es radiometrica? ¿se puede fijar paleta white-hot? ¿llega el raw o solo video comprimido?

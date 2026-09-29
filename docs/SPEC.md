# SPEC: framework de entrenamiento YOLO (RGB + IR)

Estado: **F0 hecho**. Implementar F1 a F5. F6 (hotspots / preprocesado de camara) esta **fuera de alcance**.
Fuente de contexto: `outdated/hola.ipynb`, `outdated/mot_yolo.ipynb` (codigo viejo, gitignored, solo referencia).

---

## 0. REGLA PARA EL AGENTE: NO EJECUTAR NADA REMOTO

- **Prohibido** conectarse a Colab, ejecutar notebooks, seleccionar/arrancar kernels, o cualquier cosa que consuma
  creditos o GPU. Prohibido tambien descargar datasets o pesos (Kaggle, Drive, `yolo*.pt`).
- **Prohibido** entrenar o hacer inferencia con Ultralytics, ni siquiera en local.
- Lo unico ejecutable: `python tests/test_basic.py` en local (sin GPU, sin red) y chequeos estaticos
  (`python -m py_compile`, validar que los `.ipynb` son JSON validos con `nbformat` si esta instalado).
- Los notebooks se entregan **sin outputs y sin ejecutar**. El usuario los verifica celda a celda en Colab.
- **Prohibido editar o regenerar un `.ipynb` que ya existe.** El usuario lo tiene abierto en VS Code: si el agente
  escribe el archivo en disco, se pierden las ediciones sin guardar o queda una version vieja hasta reabrirlo.
  Para cambiar un notebook existente: indicar en el chat que celda y el contenido nuevo; el usuario lo pega.
  Solo se pueden crear notebooks que todavia no existen; despues de crearlos pasan a ser del usuario.
- **Prioridad: el codigo de `src/`** (es lo que se revisa). Los notebooks son delgados: parametros + llamadas + un markdown corto por seccion.
- Todo lo que no se puede comprobar sin Colab va a la **checklist de verificacion del usuario** (seccion 7.1),
  y en el codigo se deja robusto (mensajes de error claros) en vez de asumir.

## 1. Contexto y restricciones

- El usuario edita en **VS Code (Windows)**. Los notebooks corren en un **kernel de Colab** (extension de Colab para VS Code).
  El kernel vive en la VM de Colab: **no ve archivos locales**. El codigo `src/` llega a la VM via `git pull` desde
  `https://github.com/sbstn-sss/deteccion_drones.git` (rama `main`).
- Datasets y pesos viven en **Google Drive** (montado en `/content/drive`) o se bajan de **Kaggle**. Nunca en git.
- Se descomprime siempre a disco local de la VM (`/content/datasets/...`), nunca se entrena leyendo imagenes desde Drive (lento).
- Los runs (pesos, metricas) se escriben **directo en Drive** para sobrevivir a desconexiones.
- Colab es efimero: toda operacion de setup debe ser **idempotente** (si ya esta hecho, no hace nada).
- Los notebooks no contienen logica: solo parametros, llamadas a `deteccion.*` y graficos. Celda de mas de ~15 lineas = mover a `src/`.
- **"Restart & Run All" debe funcionar** en cada notebook editando solo la celda de parametros.

### Convenciones de codigo
- Python >= 3.10, `pathlib.Path`, type hints en funciones publicas.
- Identificadores en ingles; docstrings, comentarios y prints en espanol, cortos. Sin emojis.
- `ultralytics` se importa **solo dentro de `experiment.py`** (y dentro de funciones de `viz.py` que lo necesiten), para que
  `data.py`, `env.py` y los tests corran en local sin torch.
- Nada de rutas hardcodeadas en `src/`: todo entra por argumentos.
- Una sola clase (`Experiment`) + un dataclass (`ExperimentConfig`). Lo demas son funciones.
- Trabajar en rama `feat/framework`, un commit por fase. No hacer push ni merge a `main` sin que el usuario lo pida.

---

## 2. Estructura final

```
deteccion/
├── pyproject.toml              # ya existe (F0)
├── .gitattributes              # nbstripout (F0)
├── docs/SPEC.md                # este archivo
├── src/deteccion/
│   ├── __init__.py             # re-exporta la API publica (ver 3.5)
│   ├── env.py
│   ├── data.py
│   ├── viz.py
│   ├── experiment.py           # ExperimentConfig + Experiment + list_runs
│   └── converters/
│       ├── __init__.py
│       └── visdrone_mot.py
├── notebooks/
│   ├── rgb/
│   │   ├── 00_preparar_mot.ipynb
│   │   ├── 01_explorar.ipynb
│   │   ├── 02_entrenar.ipynb
│   │   └── 03_testing.ipynb
│   └── ir/
│       ├── 01_explorar.ipynb
│       ├── 02_entrenar.ipynb
│       └── 03_testing.ipynb
└── tests/
    └── test_basic.py           # asserts planos, `python tests/test_basic.py`, sin GPU ni dataset
```

---

## 3. Modulos

### 3.1 `env.py` — entorno Colab / Drive / datasets

```python
IN_COLAB: bool                     # "google.colab" in sys.modules
DRIVE_ROOT = Path("/content/drive")
DATA_ROOT  = Path("/content/datasets")   # destino local de todos los datasets

def mount_drive() -> None
    # Si IN_COLAB y /content/drive/MyDrive no existe -> google.colab.drive.mount. Fuera de Colab: no hace nada.

def drive_path(p: str | Path) -> Path
    # Absoluta -> tal cual. Relativa (ej "MyDrive/x") -> DRIVE_ROOT / p.

def unzip_once(zip_path: str | Path, dest: str | Path) -> Path
    # Idempotente via marcador: si dest/.done existe -> no hace nada. Si no, extrae con zipfile, printea tiempo
    # y escribe dest/.done AL FINAL. (Si Colab se corta a mitad, no hay .done y se re-extrae: no queda un dataset a medias.)
    # FileNotFoundError claro si el zip no existe. Devuelve dest.

def kaggle_download_once(slug: str, dest: str | Path) -> Path
    # Mismo marcador dest/.done que unzip_once.
    # Credenciales: en Colab leer KAGGLE_USERNAME y KAGGLE_KEY de google.colab.userdata (Secrets) y ponerlas en os.environ.
    # Descarga con la API de kaggle (pip extra "kaggle", instalarlo si falta) y descomprime en dest.

def fix_data_yaml(yaml_path: str | Path, root: str | Path | None = None) -> Path
    # Lee con yaml.safe_load. Pone path = root o la carpeta del yaml. Escribe "<stem>_local.yaml" en la misma
    # carpeta (NO modifica el original). Devuelve la ruta nueva. Reemplaza la celda 7 de hola.ipynb.
```

### 3.2 `data.py` — indexar y explorar un dataset YOLO (sin leer imagenes)

Convencion Ultralytics: label de `.../images/x.jpg` es `.../labels/x.txt` (reemplazar el ULTIMO segmento `images` por `labels`).

```python
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_data_yaml(yaml_path) -> dict
    # Devuelve {"root": Path, "names": {int: str}, "splits": {"train": Path, "val": Path, "test": Path|None}}
    # Resuelve rutas relativas contra "path". "names" puede venir como lista o dict -> normalizar a dict.
    # Si un split es un .txt con lista de imagenes o una lista de carpetas: NotImplementedError con mensaje claro.

def index_split(data: dict, split: str) -> pd.DataFrame
    # Una fila por imagen: split, image_path, label_path, has_label. Orden estable (sorted).

def load_boxes(images: pd.DataFrame, names: dict) -> tuple[pd.DataFrame, pd.DataFrame]
    # Lee todos los .txt. Devuelve (boxes, issues).
    # boxes: image_path, split, cls, name, xc, yc, w, h  (normalizados)
    # issues: image_path, line_no, line, reason  -> reason en {"n_cols", "no_numerico", "fuera_de_[0,1]", "clase_desconocida", "w_o_h_cero"}
    # Lineas invalidas NO se descartan en silencio: van a issues (bug #6 del notebook viejo).

def add_image_sizes(images: pd.DataFrame, sample: int | None = None, seed: int = 0) -> pd.DataFrame
    # Agrega columnas width, height leyendo SOLO el header con PIL.Image.open(p).size.
    # sample: si se da, solo a una muestra (datasets grandes).

def add_pixel_sizes(boxes, images) -> pd.DataFrame
    # Merge con width/height -> w_px, h_px, area_px, size_bucket en {"small" (<32^2), "medium" (<96^2), "large"} (criterio COCO).
    # Cajas cuya imagen no tiene tamano (fuera del sample de add_image_sizes) se DESCARTAN, nunca se clasifican.

def summary(images, boxes, issues) -> pd.DataFrame
    # Una fila por split: n_images, n_sin_label, n_boxes, boxes_por_img_media, boxes_por_img_max, n_issues.

def class_counts(boxes) -> pd.DataFrame
    # Filas: clase; columnas: splits; valores: n_boxes. Incluye % del total de train.
```

### 3.3 `viz.py` — graficos (matplotlib + cv2), cada funcion devuelve la `Figure`

```python
def draw_boxes(img_bgr, boxes: pd.DataFrame | np.ndarray, names: dict, color=None, thickness=2) -> np.ndarray
    # boxes normalizados xc,yc,w,h (+cls). Dibuja caja + nombre de clase. Color por clase (paleta fija) si color=None.
    # Unica implementacion del dibujo de GT (bug #6: estaba duplicado).

def show_samples(images, boxes, names, n=6, seed=0, cols=3) -> Figure
    # Muestra n imagenes al azar (con seed) con sus GT. Titulo: nombre + n objetos.

def plot_class_counts(boxes, names) -> Figure              # barras por clase y split
def plot_box_sizes(boxes_px) -> Figure                     # hist 2D w_px vs h_px (log) + barras small/medium/large
def plot_boxes_per_image(boxes, images) -> Figure          # histograma de n cajas por imagen
def plot_image_sizes(images) -> Figure                     # conteo de resoluciones distintas (ignora filas sin width/height)

def show_run_plots(run_dir) -> None
    # Muestra los png que Ultralytics YA genera en el run: results.png, confusion_matrix_normalized.png,
    # BoxPR_curve.png (el nombre cambio entre versiones: buscar "*PR_curve.png"). No re-graficar results.csv.

def show_pred_vs_gt(model, images, boxes, names, n=3, seed=0, conf=0.25, imgsz=640) -> Figure
    # 2 filas x n: fila 1 prediccion (results[0].plot()), fila 2 GT con draw_boxes. Titulos con conteos.
```

### 3.4 `experiment.py`

```python
@dataclass
class ExperimentConfig:
    data_yaml: str                 # yaml ya corregido (salida de env.fix_data_yaml)
    out_dir: str                   # carpeta de runs en Drive (absoluta o relativa a DRIVE_ROOT)
    model: str = "yolo11s.pt"      # cualquier peso de Ultralytics: yolov8n/s/m.pt, yolo11s.pt, o ruta a un .pt propio
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    seed: int = 0
    tag: str = ""                  # texto libre para el nombre del run, ej "cosLR"
    train_args: dict = field(default_factory=dict)   # se pasa TAL CUAL a YOLO.train (cos_lr, patience, close_mosaic, classes, mosaic, ...)

    def run_name(self, stamp: str) -> str
        # "{dataset}_{modelo}_{imgsz}[_{tag}]_{stamp}"  ej "visdrone_yolo11s_1024_cosLR_20260925-1530"
        # dataset = stem del yaml sin "_local"; modelo = stem del .pt.
```

```python
class Experiment:
    def __init__(self, cfg: ExperimentConfig, run_name: str | None = None)
        # run_name None -> cfg.run_name(timestamp actual "%Y%m%d-%H%M"). self.run_dir = out_dir / run_name.
        # NO carga el modelo aqui.

    run_dir: Path
    best: Path              # property: run_dir/weights/best.pt
    last: Path              # property: run_dir/weights/last.pt
    model                   # property lazy: YOLO(best) si existe, si no error claro "este run no tiene best.pt"

    def train(self)
        # YOLO(cfg.model).train(data=cfg.data_yaml, epochs, imgsz, batch, seed,
        #                       project=str(out_dir), name=run_name, exist_ok=True, **cfg.train_args)
        # Antes de entrenar guarda run_dir/experiment.json (asdict(cfg)).
        # Nunca pasar augment=True (bug #8: es TTA de prediccion).
        # Al terminar, invalida el cache de self.model. Devuelve los results de Ultralytics.

    def resume(self)
        # YOLO(self.last).train(resume=True). Para cuando Colab se desconecta a mitad de entrenamiento.

    @classmethod
    def load(cls, run_dir, data_yaml: str | None = None) -> "Experiment"
        # Reconstruye cfg desde run_dir/experiment.json; si no existe (runs viejos), desde run_dir/args.yaml
        # (campos data, model, epochs, imgsz, batch). run_name = nombre de la carpeta.
        # data_yaml override: el yaml guardado apunta a /content/... de otra sesion.

    def val(self, split: str = "val", conf: float = 0.001) -> pd.DataFrame
        # self.model.val(data, split, imgsz, conf, project=run_dir, name=f"eval_{split}", exist_ok=True)
        # Devuelve DataFrame: fila "all" + una por clase con P, R, mAP50, mAP50-95.
        # Test solo al final (el notebook lo dice en markdown).

    def show_pred_vs_gt(self, split="test", n=3, seed=0, conf=0.25) -> Figure
        # Usa data.load_data_yaml/index_split/load_boxes + viz.show_pred_vs_gt.

    def predict_video(self, video, conf=0.25, **kw) -> Path
    def track(self, video, tracker="botsort.yaml", conf=0.15, iou=0.5, **kw) -> Path
        # video: ruta (drive_path aplica). Guarda en run_dir/"videos"/<stem_video>_{predict|track}.
        # track: stream=True, persist=True, consumir el generador, print cada 30 frames con n ids.
        # Devuelve la carpeta de salida. El modelo se carga UNA vez (bug #10).

def list_runs(out_dir, sort: str = "date") -> pd.DataFrame
    # Una fila por subcarpeta con args.yaml: name, path, model, data, imgsz, epochs, epochs_done,
    # mAP50, mAP50-95 (max sobre results.csv), has_best, date (mtime).
    # Columnas de results.csv: strip() de espacios (versiones viejas de Ultralytics los traen);
    # buscar "metrics/mAP50(B)" y "metrics/mAP50-95(B)". Runs sin results.csv -> NaN, no error.
    # sort in {"date", "mAP50", "mAP50-95"}, descendente.
    # Debe funcionar con los runs viejos de MyDrive/VisDrone_Prototipo (ej "yolo11s_visdrone_tracking_opt-8").
```

### 3.5 `__init__.py`

```python
from deteccion import env, data, viz
from deteccion.experiment import ExperimentConfig, Experiment, list_runs
```
Ojo: esto importa ultralytics al hacer `import deteccion`. Aceptable (en Colab siempre esta). Los tests importan
`deteccion.data` / `deteccion.env` / `deteccion.converters.visdrone_mot` directo; si el import de ultralytics en
`__init__` rompe los tests locales, hacer el import de `experiment` perezoso con `__getattr__` a nivel de modulo.

### 3.6 `converters/visdrone_mot.py` (reemplaza `outdated/mot_yolo.ipynb`)

```python
VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
                  "tricycle", "awning-tricycle", "bus", "motor"]

def convert_visdrone_mot(raw_split_dir, out_dir, split: str) -> int
    # raw_split_dir contiene sequences/ y annotations/ (si el zip crea una subcarpeta extra, bajar un nivel,
    # igual que el notebook viejo). Salida: out_dir/images/{split}/{seq}_{frame}.jpg y labels/{split}/...txt
    # Formato de linea de anotacion: frame,target_id,left,top,w,h,score,category,truncation,occlusion
    # Reglas:
    #   - score == 0 -> ignorar (region ignorada; bug #12)
    #   - category 0 (ignored) y 11 (others) -> ignorar; cls = category - 1
    #   - recortar la caja a los bordes de la imagen; descartar si w<=0 o h<=0 tras recortar
    #   - tamano de imagen: leer UNA vez por secuencia (PIL, header)
    #   - copiar imagen con shutil.copy2, NO imread/imwrite (bug #11)
    #   - frame sin objetos -> .txt vacio (background)
    # Devuelve n imagenes escritas.

def build_mot_dataset(raw_root, out_dir, splits=("train", "val")) -> Path
    # Llama convert por split y escribe out_dir/visdrone_mot.yaml (path, train, val, names). Idempotente:
    # si el yaml existe, no hace nada. Devuelve el yaml.
```
Nota para el notebook: frames consecutivos estan muy correlacionados; respetar el split oficial por secuencia, no re-mezclar.

---

## 4. Celda de setup (identica en todos los notebooks)

`import deteccion` no puede ir antes de clonar. Esta celda va en el notebook (unica excepcion a "sin logica en notebooks"):

```python
REPO = "https://github.com/sbstn-sss/deteccion_drones.git"
REPO_DIR = "/content/deteccion_drones"
BRANCH = "main"   # cambiar para probar una rama antes del merge, ej "feat/framework"

import sys
if "google.colab" in sys.modules:
    !git -C {REPO_DIR} checkout -q {BRANCH} 2>/dev/null && git -C {REPO_DIR} pull -q 2>/dev/null || git clone -q -b {BRANCH} {REPO} {REPO_DIR}
    !pip install -q -e {REPO_DIR}
    sys.path.insert(0, f"{REPO_DIR}/src")   # pip -e usa un .pth que el kernel ya iniciado no lee: sin esto falla el import

try:  # el IPython de Colab con Python 3.13 trae un autoreload que importa 'imp' (eliminado en 3.12)
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except ModuleNotFoundError:
    print("autoreload no disponible: reinicia el kernel tras editar src/")
from deteccion import env, data, viz
from deteccion import ExperimentConfig, Experiment, list_runs   # solo en 02/03 (01 y 00 no la llevan)
env.mount_drive()
```

Verificado por el usuario en Colab desde VS Code (2026-09-28). Notas:
- El repo es **publico**: el clone no usa credenciales. No usar Colab Secrets (`google.colab.userdata`) para esto:
  no funciona de forma confiable desde la extension de VS Code.
- La rama (`BRANCH`) tiene que estar pusheada: Colab clona desde GitHub, no ve el disco local.

**Riesgo (lo verifica el usuario, checklist 7.1):** que `drive.mount` funcione con la extension de Colab para VS Code.
En `env.mount_drive()`: si el mount lanza excepcion, re-lanzarla con un mensaje que diga que probar el notebook
en Colab web (abrir desde GitHub). No implementar alternativas.

---

## 5. Notebooks (celda por celda)

Todos: celda 1 markdown (titulo + que hace / que NO hace), celda 2 **PARAMETROS**, celda 3 setup (seccion 4), luego pasos.
Ninguna ruta literal fuera de la celda de parametros. Cada seccion con un markdown de 1-2 lineas explicando el porque.

### Parametros por dominio

**Las rutas de abajo son de ejemplo, no las reales.** El usuario reorganizo las carpetas en Drive y edita a mano la
celda de parametros de cada notebook. En notebooks nuevos usar estas rutas como placeholder; nunca "corregir" las
que el usuario puso en un notebook para que coincidan con el spec. Por eso ninguna ruta puede aparecer fuera de la
celda de parametros.

```python
# rgb (VisDrone DET)
DATASET_ZIP = "MyDrive/deteccion_drones_v1/archive.zip"
DATASET_DIR = "visdrone"                           # subcarpeta en env.DATA_ROOT
DATA_YAML   = "VisDrone_Dataset/visdrone.yaml"     # relativo a DATASET_DIR
OUT_DIR     = "MyDrive/deteccion_drones/runs/rgb"
MODEL       = "yolo11s.pt"
EPOCHS, IMGSZ, BATCH = 50, 1024, 8
TRAIN_ARGS  = dict(cos_lr=True, close_mosaic=15, patience=10, workers=2)

# ir (HIT-UAV)
DATASET_ZIP = "MyDrive/deteccion_drones/hit-uav.zip"  # bajado desde la web de Kaggle y subido a Drive (ver nota)
DATASET_DIR = "hituav"
DATA_YAML   = "hit-uav/dataset.yaml"               # VERIFICAR tras descargar
OUT_DIR     = "MyDrive/deteccion_drones/runs/ir"
MODEL       = "yolo11s.pt"                         # decidido 2026-09-28: misma precision que v8s con ~25% menos GFLOPs
EPOCHS, IMGSZ, BATCH = 100, 640, 16                # imagenes 640x512
TRAIN_ARGS  = dict(patience=20, workers=2)         # + classes=[...] segun decision DontCare
```

### `01_explorar` (rgb e ir, misma estructura)
1. Obtener dataset: `unzip_once` (rgb e ir) -> `fix_data_yaml`.
   Nota IR: se usa zip en Drive y no `kaggle_download_once` porque `google.colab.userdata` (Secrets) no es confiable
   desde la extension de VS Code (verificado por el usuario en rgb/01). `kaggle_download_once` queda en `env.py` sin uso;
   si se retoma, debe aceptar credenciales por variables de entorno cuando `userdata.get` falle.
2. `data = load_data_yaml`, `images = concat(index_split(s) for s in splits)`, `boxes, issues = load_boxes`.
3. `summary(...)` + `issues.head()` (si hay issues, markdown explicando que hacer).
4. `class_counts` + `plot_class_counts`.
5. `add_image_sizes(sample=2000)` -> `plot_image_sizes`, `add_pixel_sizes` -> `plot_box_sizes`.
6. `plot_boxes_per_image`.
7. `show_samples(n=6, seed=0)`.
8. Markdown final "Conclusiones" vacio para que el usuario anote (ej: % de objetos small justifica imgsz).
- **ir extra:** confirmar que las labels ya vienen en YOLO (si no: parar y reportar, hara falta converter).
  Mostrar conteo de `DontCare` y su indice de clase; markdown con la decision pendiente
  (opcion por defecto: excluirla con `TRAIN_ARGS["classes"] = [indices sin DontCare]`, filtro nativo de Ultralytics).
  Confirmar que las imagenes son 1 canal/gris (YOLO las replica a 3 canales solo, no convertir).

### `02_entrenar`
1. Obtener dataset (igual que 01, idempotente) + `fix_data_yaml`.
2. `cfg = ExperimentConfig(...)`, `exp = Experiment(cfg)`, print `exp.run_dir`.
3. `exp.train()`.
4. `viz.show_run_plots(exp.run_dir)` + `exp.val("val")`.
5. Celda comentada: `# exp = Experiment.load(RUN_DIR); exp.resume()` para retomar tras desconexion.

### `03_testing`
Parametros extra: `RUN = None` (None -> el de mejor mAP50-95), `VIDEO = "MyDrive/deteccion_drones_v1/videos/video1_testing.mp4"` (rgb; en ir `None` hasta tener video).
1. Obtener dataset + `fix_data_yaml`.
2. `runs = list_runs(OUT_DIR, sort="mAP50-95")`; mostrar tabla.
3. `exp = Experiment.load(runs.path.iloc[0] if RUN is None else f"{OUT_DIR}/{RUN}", data_yaml=...)`.
4. `exp.val("test")` (solo si el yaml tiene test).
5. `exp.show_pred_vs_gt(n=3, seed=0)`.
6. Si `VIDEO`: `exp.predict_video(VIDEO)` y `exp.track(VIDEO)`.

### `rgb/00_preparar_mot`
Parametros: `MOT_ZIPS = {"train": "MyDrive/deteccion_drones_v1/motdataset/VisDrone2019-MOT-train.zip", "val": ".../VisDrone2019-MOT-val.zip"}`, `MOT_OUT_ZIP = "MyDrive/deteccion_drones_v1/visdrone_mot_yolo.zip"`.
1. Si `MOT_OUT_ZIP` existe -> print y terminar (ya esta hecho).
2. `unzip_once` de cada zip -> `build_mot_dataset` -> `shutil.make_archive` a `MOT_OUT_ZIP`.
3. `show_samples` de 6 frames convertidos para verificar a ojo.
Luego se entrena con `rgb/02_entrenar` cambiando `DATASET_ZIP`/`DATA_YAML`.

---

## 6. Tests (`tests/test_basic.py`)

Asserts planos, ejecutable con `python tests/test_basic.py` en local (sin GPU, sin dataset, sin ultralytics). Crea datos sinteticos en un `tempfile.TemporaryDirectory`:
1. `load_boxes`: un .txt con 1 linea valida + lineas malas (4 columnas, coordenada 1.3, clase 99, texto) -> 1 box y 4 issues con el reason correcto.
2. `index_split`: label_path correcto aunque la ruta tenga "images" en un directorio padre.
3. `fix_data_yaml`: escribe `_local.yaml` con path correcto y no toca el original.
4. `ExperimentConfig.run_name`: formato exacto con y sin tag, stem sin "_local".
5. `convert_visdrone_mot` con 1 secuencia de 2 frames (jpg de 100x50 generado con PIL): score 0 ignorado, categoria 0 y 11 ignoradas, caja recortada al borde, frame sin objetos -> txt vacio, coordenadas normalizadas correctas.

---

## 7. Fases y criterio de aceptacion

Aceptacion del agente = codigo escrito + `python tests/test_basic.py` pasa + `py_compile` limpio + notebooks JSON validos sin outputs.
Nada se ejecuta en Colab (seccion 0).

| Fase | Entregable |
|------|-----------|
| F1 | `env.py`, `data.py`, `viz.py`, tests 1-3, `rgb/01_explorar` |
| F2 | `experiment.py` (config, train, resume, load, val, list_runs), test 4, `rgb/02_entrenar` |
| F3 | `show_pred_vs_gt`, `predict_video`, `track`, `rgb/03_testing` |
| F4 | `converters/visdrone_mot.py`, test 5, `rgb/00_preparar_mot` |
| F5 | `ir/01_explorar`, `ir/02_entrenar`, `ir/03_testing` (idealmente sin tocar `src/`; si hace falta, justificarlo) |

Al terminar cada fase: commit en `feat/framework` y un resumen corto (que se hizo, que tests pasan, que queda en la checklist 7.1).

### 7.1 Checklist de verificacion del USUARIO (en Colab, cuando el decida)

El agente deja esta lista copiada al final de su resumen, marcando que items agrego o toco:
- [ ] Celda de setup: clone/pull + import funcionan desde VS Code con kernel Colab.
- [ ] `drive.mount` funciona desde la extension de VS Code (si no: usar Colab web).
- [ ] `rgb/01_explorar`: Restart & Run All sin editar nada salvo parametros; estructura del zip de VisDrone coincide con `DATA_YAML`.
- [ ] `rgb/02_entrenar` con `EPOCHS=1` y `TRAIN_ARGS["fraction"]=0.05` (barato): escribe el run en Drive.
- [ ] `list_runs` muestra ese run junto a los viejos de `MyDrive/VisDrone_Prototipo`.
- [ ] `rgb/03_testing` con `RUN="yolo11s_visdrone_tracking_opt-8"`: mAP test, pred vs GT, video trackeado.
- [ ] `rgb/00_preparar_mot`: zip YOLO del MOT en Drive.
- [ ] `ir/01_explorar`: slug de Kaggle, ruta del yaml, labels en YOLO, indice de `DontCare`, imagenes en gris.
- [ ] `ir/02_entrenar` corto: `classes=[...]` filtra DontCare en la version instalada.

---

## 8. Fuera de alcance / pendiente

- F6: `thermal/hotspots.py` y `thermal/preprocess.py`. Sin datos de camara todavia. Notas en `investigacion/plan_refactor.md`.
- Benchmark de latencia para el outpost (PENDIENTE: el usuario debe confirmar el hardware).
  Outpost/lab = PC con GPU dedicada (modelo por confirmar). Idea: script que el usuario corre EN ESE PC (no Colab)
  midiendo ms/frame con pesos preentrenados, batch 1, grilla modelo {n,s,m} x imgsz {640,800,1024}, en el formato de
  despliegue (TensorRT/ONNX FP16). La latencia no depende del entrenamiento: se mide antes y se entrena solo lo que cabe.
  Preguntas abiertas: modelo de GPU, fps de deteccion requeridos, procesos que comparten el equipo.
- W&B: no integrar. Ultralytics ya guarda `args.yaml` + `results.csv` + plots por run.
- Export (ONNX/TensorRT), despliegue en dron.
- Decision `DontCare` en HIT-UAV: la toma el usuario tras ver `ir/01_explorar`.

## 9. Supuestos no verificados (el agente NO los comprueba; van a la checklist 7.1, el codigo falla con mensaje claro si no se cumplen)

- Slug de Kaggle de HIT-UAV y ruta de su `dataset.yaml`; que las labels vengan en YOLO; indice de `DontCare`.
- Estructura exacta del zip de VisDrone DET (`VisDrone_Dataset/visdrone.yaml` segun el notebook viejo) y claves del yaml.
- `drive.mount` bajo la extension de Colab para VS Code.
- Que `classes=[...]` en `train()` filtre labels en la version de Ultralytics instalada.
- Nombres de archivos de plots en el run (cambian entre versiones de Ultralytics).

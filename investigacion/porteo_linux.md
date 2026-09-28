---
title: "Guia de porteo a Linux: del framework en Colab al pipeline ROS2 + YOLO + Gazebo"
fecha: 2026-09-28
estado: guia (pendiente de ejecutar)
---

# Guia de porteo a Linux

Para leer antes de migrar. Resume donde quedo el proyecto en Windows/Colab, que hay que llevarse,
que cambia en Linux y en que orden atacar la etapa ROS2 + Gazebo.

---

## 0. Donde quedo todo (estado al 2026-09-28)

**Codigo** (rama `feat/framework`, repo `sbstn-sss/deteccion_drones`):

| Pieza | Estado |
|-------|--------|
| `src/deteccion/env.py`, `data.py`, `viz.py` | hecho (F1) |
| `src/deteccion/experiment.py`: `ExperimentConfig`, `Experiment` (train/resume/load/val/predict/track), `list_runs` | hecho (F2, F3) |
| `notebooks/rgb/01-03`, `notebooks/ir/01-03` | hechos y corridos en Colab |
| `converters/visdrone_mot.py` + `rgb/00_preparar_mot` | **NO implementado** (F4 pendiente) |
| `tests/test_basic.py` | pasa local, sin GPU |

**Datos y modelos** (viven en Google Drive, no en git):

| Que | Donde |
|-----|-------|
| VisDrone DET | `MyDrive/deteccion_drones/datasets/visdrone.zip` |
| HIT-UAV | `MyDrive/deteccion_drones/hit-uav.zip` |
| Runs RGB | `MyDrive/deteccion_drones/runs/rgb/` |
| Runs IR | `MyDrive/deteccion_drones/runs/ir/` |
| Runs viejos | `MyDrive/VisDrone_Prototipo/` (ej. `yolo11s_visdrone_tracking_opt-8`) |
| Videos de prueba | `MyDrive/deteccion_drones/videos/` |

**Lo que se sabe de los datos:**
- HIT-UAV: labels YOLO, gris 1 canal, 640x512, `DontCare` = indice 4 (se excluye con `classes=[0,1,2,3]`),
  **99.7% de las personas son small** y ninguna es large.
- VisDrone: personas en dos clases (`pedestrian`, `people`) -> ojo con duplicados al fusionarlas.

Notas relacionadas: [capa_alertas.md](capa_alertas.md), [plan_refactor.md](plan_refactor.md), spec en [`../docs/SPEC.md`](../docs/SPEC.md).

---

## 1. Antes de apagar Windows (checklist)

- [ ] Terminar los entrenamientos que esten corriendo y elegir **un modelo RGB y uno IR** (por recall de persona en val).
- [ ] `03_testing` de cada uno: anotar mAP test y recall de persona en las Conclusiones del notebook.
- [ ] Commitear los notebooks `rgb/` (tienen cambios de parametros sin subir). nbstripout limpia los outputs.
- [ ] Merge de `feat/framework` a `main` y push. En Linux se trabaja desde `main`.
- [ ] Descargar de Drive la carpeta completa de los runs elegidos (no solo `best.pt`: tambien `args.yaml`,
      `experiment.json`, `results.csv`). Guardarlos fuera de git (los `.pt` estan en `.gitignore`).
- [ ] Anotar en una tabla: run elegido, modelo, imgsz, clases, metricas. Es lo que el nodo detector va a cargar.

---

## 2. Entorno Linux

### Versiones (confirmar con el pipeline base heredado)

Las combinaciones oficiales que conozco (verificar contra lo que use el pipeline que te pasaron):

| Ubuntu | ROS2 | Gazebo |
|--------|------|--------|
| 22.04 | Humble | Fortress (o Harmonic con paquetes extra) |
| 24.04 | Jazzy | Harmonic |

**Regla:** no cambiar la version del pipeline base al portear. Adaptar el framework a ese entorno, no al reves.

### Python con ROS2

ROS2 usa el Python del sistema. Para instalar ultralytics/torch sin romper ROS2:

```bash
python3 -m venv --system-site-packages ~/venvs/deteccion   # ve rclpy, cv_bridge, etc.
source /opt/ros/<distro>/setup.bash
source ~/venvs/deteccion/bin/activate
pip install -e ~/deteccion_drones            # el framework
python tests/test_basic.py                    # primer chequeo
```

Riesgos conocidos a revisar:
- **numpy 2 vs cv_bridge**: en ROS2 Humble, `cv_bridge` esta compilado contra numpy 1.x; si ultralytics instala
  numpy 2, `cv_bridge` puede fallar al importar. Solucion habitual: `pip install "numpy<2"`. Verificar en tu distro.
- **torch con CUDA**: instalar la version de torch que coincida con el driver/CUDA de la GPU del laboratorio
  (instrucciones en pytorch.org). Chequeo: `python -c "import torch; print(torch.cuda.is_available())"`.
- **opencv duplicado**: ultralytics trae `opencv-python`; ROS2 trae el de sistema. Si hay conflictos de GUI
  (`cv2.imshow`), usar `opencv-python-headless` en el venv.

---

## 3. Cambios necesarios en el framework

### 3.1 Rutas fijas de Colab en `env.py` (OBLIGATORIO)

Hoy: `DRIVE_ROOT = /content/drive`, `DATA_ROOT = /content/datasets`, y `drive_path("MyDrive/x")` asume Colab.
En Linux esas rutas no existen.

Cambio chico: leer ambas desde variables de entorno con el valor de Colab por defecto.

```python
DRIVE_ROOT = Path(os.environ.get("DETECCION_DRIVE_ROOT", "/content/drive"))
DATA_ROOT  = Path(os.environ.get("DETECCION_DATA_ROOT", "/content/datasets"))
```

En Linux, por ejemplo: `DETECCION_DRIVE_ROOT=~/drive` (una copia local de las carpetas de Drive, o Drive montado con rclone)
y `DETECCION_DATA_ROOT=~/datasets`. Asi los notebooks y `list_runs` funcionan igual sin tocar sus parametros.

### 3.2 Notebooks fuera de Colab

- La celda de setup (git clone/pull + `pip install -e`) solo corre si esta en Colab: en Linux se salta sola.
  Basta con haber hecho `pip install -e .` en el venv y seleccionar ese kernel.
- `env.mount_drive()` no hace nada fuera de Colab (ya es asi).
- Entrenar en el PC del laboratorio es posible si la GPU aguanta (ver seccion 6): mismos notebooks `02_entrenar`.

### 3.3 Pendientes del framework que se llevan a Linux

- [ ] F4: convertidor VisDrone-MOT (spec seccion 3.6). Solo si se decide entrenar con MOT.
- [ ] `botsort_dron.yaml` en el repo (tracker ajustado: `track_buffer: 90`, `new_track_thresh: 0.4`, `gmc_method: sparseOptFlow`).
- [ ] Mapeo de clases al contrato (`CLASS_MAP`: pedestrian/people/Person -> `person`) + NMS despues de mapear
      (o `agnostic_nms=True`) para no duplicar personas de VisDrone.
- [ ] Experimentos anotados, no urgentes: `imgsz` 640 vs 1024 (IR `hires`), `scale=0.3`, `freeze=10`,
      remapear clases a indices COCO para heredar la neurona `person`, set de prueba de corta distancia.

---

## 4. Del notebook al nodo ROS2

El framework (`src/deteccion`) es para **entrenar y evaluar**. En ROS2 solo se necesita **cargar el `.pt` e inferir**.
No hace falta meter `Experiment` en el nodo.

Arquitectura objetivo (detalle de la ultima etapa en [capa_alertas.md](capa_alertas.md)):

```
camara (Gazebo o real) --image--> [detector] --detecciones--> [georef] --puntos en DEM--> [alertas] --> orquestador
                                      ^                           ^
                              best.pt RGB o IR             pose del dron + tf + DEM 12.5 m
```

Nodo `detector` (partir del que trae el pipeline base, no reescribir):
- Parametros ROS: ruta al `.pt`, `imgsz`, `conf`, `classes`, `sensor` (rgb/ir). El toggle dia/noche = cambiar de `.pt`.
- Publicar solo `person` (contrato acotado) con bbox, conf, timestamp, sensor.
- El `imgsz` de inferencia debe ser el mismo del entrenamiento.
- Si el pipeline usa `vision_msgs/Detection2DArray`, usarlo: evita inventar mensajes.

Nodo `georef`: ray casting del **centro inferior** de cada bbox al DEM (Python), con la pose del dron via `tf2`.
Tambien proyectar las 4 esquinas de la imagen: footprint de la camara (lo necesita la grilla de alertas).

Nodo `alertas`: grilla de evidencia + zonas + maquina de estados (ver capa_alertas.md).

---

## 5. Simulacion en Gazebo

### Lo que hay que construir para el caso concreto
- Mundo con terreno a partir del **DEM real** (heightmap) para que el ray casting de la simulacion y el real coincidan.
- Personas como **actors** con trayectorias (caminar, detenerse, grupos) y vehiculos si aplica.
- Dron con camara RGB (y termica, si se puede: Gazebo tiene sensor de camara termica; verificar soporte
  en la version del pipeline y como se asigna temperatura a los actors).
- Bridge `ros_gz` para llevar imagen, pose y ground truth de los actors a ROS2.

### Riesgo importante: la brecha simulacion -> modelo
Los YOLO se entrenaron con imagenes **reales** (VisDrone, HIT-UAV). Las personas renderizadas por Gazebo se ven distintas:
es posible que el detector rinda mucho peor en simulacion que en la realidad. Consecuencias:
- No concluir "el detector no sirve" por lo que pase en Gazebo.
- Para probar georref y alertas sin depender del detector: **inyectar detecciones sinteticas** desde el ground truth
  de Gazebo (posicion real de los actors -> proyectar a la camara -> bbox con ruido y fallas aleatorias).
  Asi se prueban las capas de abajo con un detector "controlado", y el detector real se evalua aparte con video real.

### Ground truth gratis
Gazebo sabe donde esta cada persona: sirve para medir tiempo a alerta, falsas alertas por hora y error de posicion
(metricas de capa_alertas.md, seccion 7). Grabar todo con `ros2 bag` para reprocesar sin volver a simular.

---

## 6. Hardware del laboratorio (pendiente)

PC con GPU dedicada, modelo por confirmar. Cuando se sepa:
1. Benchmark de latencia **antes** de decidir que entrenar: grilla modelo {n, s, m} x imgsz {640, 800, 1024},
   batch 1, con pesos preentrenados (la latencia no depende del entrenamiento), en el formato de despliegue.
2. Exportar el elegido: `YOLO("best.pt").export(format="engine", half=True)` (TensorRT FP16) o `format="onnx"`.
   Medir de nuevo: la exportacion suele acelerar 2-3x.
3. Si la GPU tiene >= 12 GB, entrenar ahi en vez de Colab (sin creditos).

Preguntas para el equipo de integracion: modelo de GPU, fps de deteccion requeridos, que otros procesos comparten el PC.

---

## 7. Que aprender (en orden de uso)

1. **ROS2 basico**: nodos, topics, mensajes, parametros, launch files, `ros2 bag`. Leer el pipeline base con esto en mente.
2. **tf2**: arbol de frames (mundo -> dron -> gimbal -> camara). Es la base de la georreferencia.
3. **Modelo de camara**: intrinsecos (`sensor_msgs/CameraInfo`), pixel -> rayo en el frame de la camara.
4. **Gazebo**: mundos, heightmaps, actors, sensores, bridge `ros_gz`.
5. **Exportacion**: TensorRT/ONNX, FP16, medicion de latencia.

---

## 8. Orden sugerido de trabajo en Linux

1. Levantar el pipeline base tal cual viene y entender su flujo (sin cambios).
2. Venv + `pip install -e` del framework + tests. Aplicar el cambio de rutas de 3.1.
3. Reemplazar el modelo del nodo detector por los `.pt` propios (RGB e IR) y probar con un video real grabado.
4. Benchmark en la GPU del laboratorio y exportar a TensorRT.
5. Mundo Gazebo propio: DEM + actors + dron con camara.
6. Nodo georef (con detecciones inyectadas desde ground truth primero).
7. Nodo alertas + metricas con ground truth.
8. Recien al final: detector real dentro de la simulacion completa.

## Preguntas abiertas (todas juntas)

- [ ] Ubuntu / ROS2 / Gazebo del pipeline base.
- [ ] GPU del PC del laboratorio; fps de deteccion requeridos; procesos que comparten el equipo.
- [ ] Formato del stream termico real (raw 16-bit / gris / paleta) y si la camara es radiometrica.
- [ ] Formato y frecuencia de mensajes hacia el orquestador; latencia maxima de una alerta.
- [ ] Precision de la pose del dron (GPS/IMU/gimbal): define la incertidumbre de la georreferencia.
- [ ] ¿Gazebo de la version usada soporta camara termica?

---
title: "Plan de experimentos: fusion de clases, modelo m y costo en el outpost"
fecha: 2026-09-28
estado: plan
---

# Plan de experimentos

## Punto de partida

| Modalidad | Run | Modelo | imgsz | Resultado |
|-----------|-----|--------|-------|-----------|
| RGB | `yolo11s_visdrone_tracking_opt-8` (viejo) | yolo11s | 1024 | val: mAP50 0.497, mAP50-95 0.303, R 0.489 (todas las clases) |
| RGB | `visdrone_yolov8s_1024_20260928-1707` | yolov8s | 1024 | val: empata con el viejo (0.306). Cortar |
| IR | `dataset_yolov8s_640_base_20260928-1702` | yolov8s | 640 | **test: Person P 0.905, R 0.892, mAP50 0.928, mAP50-95 0.513** |

Decision tomada: **yolo11s** como base (misma precision que v8s con ~25% menos GFLOPs).

Techo RGB: dos arquitecturas y dos schedules distintos convergen a ~0.30 mAP50-95 / ~0.49 R.
Lo limitan resolucion, capacidad del modelo y datos, no el numero de epocas.

## Hallazgo: `cls_remap` (Ultralytics 8.4)

Log de IR: `Remapped 3/5 cls head rows from pretrained weights by class name`. Con `cls_remap=True` (default),
las filas de la cabeza de clasificacion de COCO se copian a las clases **con el mismo nombre**
(`Person`->`person`, `Car`->`car`, `Bicycle`->`bicycle`; parece no distinguir mayusculas).

- HIT-UAV ya hereda la neurona `person` de COCO.
- VisDrone NO: sus clases de persona se llaman `pedestrian` y `people`.
  Fusionarlas en una clase llamada **exactamente `person`** hereda esa neurona gratis.
- Mismo truco para vehiculos si se fusionan: nombres COCO (`car`, `truck`, `bus`, `motorcycle`, `bicycle`).

## E1. Fusion de clases de persona (RGB) — PRIMERO

Por que primero: mismo costo de computo en el outpost (no cambia modelo ni imgsz) y ataca directo la prioridad.

**Cambio de datos** (un solo cambio, para poder atribuir el resultado):
- `pedestrian` (0) + `people` (1) -> `person`. Las otras 8 clases quedan igual.
- Opcional despues, en otro experimento: `motor` -> `motorcycle` (nombre COCO, activa cls_remap).

**Implementacion en el framework** (nueva funcion en `data.py`):
`remap_dataset(src_yaml, mapping, names, out_dir) -> Path(yaml)`
- Reescribe solo los `.txt` de labels con los indices nuevos.
- Las imagenes NO se copian: `out_dir/images` es un symlink a las imagenes originales
  (Ultralytics deriva labels de la ruta de imagenes, asi que labels nuevas + symlink de imagenes funciona).
- Idempotente (marcador `.done`), igual que `unzip_once`.
- Test local con un dataset sintetico de 2 imagenes.

**Evaluacion justa** (el punto delicado): el modelo base tiene 2 clases de persona y el nuevo 1, asi que el mAP por clase
no es comparable. Hace falta una metrica **del contrato**:
`exp.contract_recall(split, class_map={"pedestrian": "person", "people": "person"}, conf=CONF_OPERATIVO)`
- Mapea las predicciones al contrato, hace NMS entre las cajas ya mapeadas (elimina duplicados pedestrian/people),
  y mide recall y precision de `person` a IoU 0.5 contra las labels fusionadas.
- Sirve para comparar CUALQUIER modelo (viejo, fusionado, m, IR) con la misma vara: es la metrica que le importa al sistema.
- Se evalua al **conf operativo** (el que se usara en el outpost), no integrado sobre todos los umbrales como mAP.

**Run:** yolo11s, 1024, ~60-80 epocas (no 250), mismos TRAIN_ARGS que el viejo, `TAG="person"`.

**Exito si:** recall de persona (contrato) > el del run viejo al mismo conf, sin perder precision.

## E2. Modelo m — SOLO si E1 no alcanza

| | yolo11s | yolo11m |
|---|---|---|
| Parametros | 9.4 M | ~20 M |
| GFLOPs @640 | 21.5 | ~68 (x3.2) |
| GFLOPs @1024 | ~55 | ~174 |
| Entrenar 1024 en T4 | ~10 min/epoca | ~25-30 min/epoca (estimado) -> usar L4/A100 |
| Entrenar 1024 en A100 | ~2.2 min/epoca | ~6 min/epoca (estimado) |

(GFLOPs de la documentacion de Ultralytics, citados de memoria: verificar.)

Alternativa mas barata que m@1024 para objetos chicos: **s@1280 ~86 GFLOPs** (la mitad que m@1024).
Orden sugerido si E1 no alcanza: s@1280 primero, m@1024 despues.

Run: sobre el dataset fusionado de E1 (partir de lo que gano), una sola variable cambiada.

## Costo en el outpost

No se puede saber en ms sin la GPU del laboratorio. Referencias para dimensionar (tabla de Ultralytics para GPU T4
con TensorRT FP16 a 640, citadas de memoria, verificar): yolo11s ~2.5 ms, yolo11m ~4.7 ms por imagen.
Escaladas a 1024 (x2.56): **s ~6-7 ms, m ~12 ms por frame** en una GPU clase T4.

Reductores reales del costo:
- Inferencia rectangular: video 16:9 a 1024 se procesa como 1024x576 (~56% del cuadrado).
- TensorRT FP16 (x2-3 vs PyTorch), INT8 (mas, midiendo perdida).
- Batching de varios drones (ver multi_uav.md).

**Benchmark en la GPU del laboratorio** (pendiente de saber cual es): grilla {s, m} x {640, 1024, 1280},
batch {1, 4, 8}, TensorRT FP16, con pesos preentrenados (la latencia no depende del entrenamiento).
Con eso se decide si E2 es viable antes de gastar en entrenarlo.

## IR: estado

Bien por ahora. Person en test: **R 0.89, P 0.90, mAP50 0.93**. No hace falta tocar nada urgente.

- mAP50-95 de Person (0.51) es mucho menor que Car (0.74): no es un problema de deteccion sino de **ajuste fino de
  cajas diminutas** (con 10 px, 1-2 px de error ya baja el IoU). Para alertas importa encontrar la persona (R, mAP50),
  no el borde exacto.
- Val tiene solo 287 imagenes: las metricas saltan +-0.05 entre epocas. Decidir con test al final o con varias epocas.
- Pendientes IR (no urgentes):
  - Re-entrenar con **yolo11s** por consistencia con RGB y eficiencia (mismos parametros, `TAG="base"`).
  - `hires` (1024) solo si el recall en video real sale bajo: ya esta en 0.89.
  - Riesgos que el dataset no mide: personas cercanas (0 large en HIT-UAV), ciclistas etiquetados como Bicycle,
    y la brecha con la camara termica real (formato pendiente).

## Orden

1. Cortar el run RGB v8s. Candidato RGB actual: `yolo11s_visdrone_tracking_opt-8`.
2. Implementar `remap_dataset` + `contract_recall` (framework) y medir el run viejo con `contract_recall`: linea base.
3. E1: entrenar yolo11s fusionado. Comparar con `contract_recall`.
4. IR: re-entrenar con yolo11s (barato, T4) y medir con `contract_recall` (Person ya es `person`).
5. Benchmark en la GPU del laboratorio.
6. E2 (s@1280 o m@1024) solo si E1 no alcanza y el benchmark lo permite.

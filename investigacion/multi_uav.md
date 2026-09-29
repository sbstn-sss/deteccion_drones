---
title: "Multiples UAV: procesar N flujos de video en el outpost"
fecha: 2026-09-28
estado: diseno
---

# Multiples UAV en el outpost

## Respuesta corta

No es una traba: es un problema resuelto. Patron: **un solo modelo en la GPU, que recibe frames de todos los drones
en lotes (batching)**. No un YOLO por dron.

## Por que no un modelo por dron

- Los pesos pesan poco (yolo11s ~20 MB), pero cada **proceso** con CUDA reserva su propio contexto (cientos de MB)
  y los procesos compiten por la GPU sin coordinarse: peor rendimiento total.
- Una instancia que procesa un lote de N imagenes aprovecha mucho mejor la GPU que N instancias con lotes de 1.

## Arquitectura

```
dron 1 --video--> [decode] --\
dron 2 --video--> [decode] ----> [detector: junta el ultimo frame de cada dron -> model([f1, f2, ..., fN])] 
dron N --video--> [decode] --/                    |
                                                  +--> detecciones etiquetadas con drone_id + timestamp + sensor
                                                          |
                                         [tracker por dron] (estado separado por drone_id, CPU barato)
                                                          |
                                         [georef por dron] (pose de ESE dron) --> [grilla de alertas UNICA]
```

- En ROS2: un nodo detector suscrito a los N topics de imagen. En cada ciclo toma el frame mas reciente de cada dron
  y hace **una** inferencia con la lista. `model(lista_de_frames)` de Ultralytics ya acepta lotes.
- Ultralytics tambien acepta varias fuentes de video a la vez (archivo `.streams` con una URL por linea).
- RGB e IR: dos modelos cargados en el mismo proceso (el de dia y el de noche). Cada frame va al modelo de su sensor.

## Dimensionamiento (la formula)

```
capacidad_GPU (frames/s) = 1000 / t_frame_en_lote (ms)
drones_max = capacidad_GPU * utilizacion_objetivo / fps_deteccion_por_dron
```

Ejemplo con referencias aproximadas (a confirmar con el benchmark real):
yolo11s @1024 TensorRT FP16 ~6-7 ms por frame -> ~150 frames/s. Al 70% de uso y 10 fps por dron: **~10 drones**.
Con yolo11m (~12 ms): ~5-6 drones. Detectar a 5 fps por dron (con tracker entre medio) duplica la cuenta.

El batching agrega unos ms de latencia (esperar a juntar el lote). Irrelevante para alertas que exigen segundos
de persistencia.

## Cuellos de botella que no son YOLO

- **Decodificar video**: N flujos H.264/H.265 en CPU puede saturar antes que la GPU. Usar decodificacion por
  hardware (NVDEC: GStreamer `nvv4l2decoder`, ffmpeg `-hwaccel cuda`).
- **Red**: N flujos x bitrate hacia el outpost.
- **Preprocesado**: redimensionar N frames en CPU. En GPU si hace falta.

## Herramientas que ya resuelven esto (si el nodo propio no alcanza)

- **NVIDIA DeepStream**: hecho para N camaras -> decode por hardware -> inferencia en lote -> tracking por stream.
  Tiene integracion con ROS2. Mas potente, curva de aprendizaje mayor.
- **NVIDIA Triton Inference Server**: servidor de modelos con batching dinamico. El nodo ROS2 le manda frames y el
  arma los lotes.
- Recomendacion: empezar con el nodo detector propio que junta frames (simple, suficiente para pocos drones).
  Migrar a Triton/DeepStream solo si el benchmark muestra que no alcanza.

## Varios outposts

- Cada outpost procesa sus drones.
- La **grilla de evidencia** (capa_alertas.md) es aditiva: si dos drones o dos outposts ven la misma celda,
  la evidencia se suma y sale **una sola alerta**. La deduplicacion entre drones viene gratis con la grilla.
- Decision abierta: grilla central (los outposts mandan evidencia por celda) o grilla por outpost + fusion en el orquestador.

## Escenario de prueba: 2 drones fisicos + N drones digitales (Gazebo)

Con 2 drones fisicos el computo no es problema (la cuenta de arriba da ~10 por GPU). Los riesgos estan en **mezclar
simulacion y realidad** en el mismo sistema:

- **Misma interfaz ROS2 para todos**: un dron digital y uno fisico publican los mismos topics (imagen, pose,
  camera_info) con el mismo tipo de mensaje. El detector, georef y alertas no deben saber cual es cual.
  Solo cambia el namespace (`/uav1/...`, `/sim_uav3/...`).
- **Relojes**: Gazebo usa tiempo simulado (`use_sim_time`), los drones fisicos tiempo real. Si se mezclan en la misma
  grilla, los timestamps no son comparables y el decaimiento de evidencia se rompe.
  Decidir: correr Gazebo en tiempo real (factor 1.0) y todo con reloj de pared, o separar grillas sim/real.
- **Mismo mundo**: para que un dron digital y uno fisico aporten a la misma grilla, el mundo de Gazebo tiene que ser
  un gemelo del terreno real (mismo DEM, mismo origen de coordenadas). Si no, son dos mapas distintos.
- **Marcar el origen**: cada deteccion y cada alerta lleva `source: sim | real`. Permite filtrar alertas sinteticas,
  y medir por separado (el detector rinde distinto en imagenes renderizadas, ver porteo_linux.md seccion 5).
- **Personas**: en la simulacion son actors; en la prueba fisica son personas reales. Una alerta "real" confirmada
  por un dron digital solo tiene sentido si el actor esta donde esta la persona (inyectado desde su GPS, por ejemplo).
  Definir que se quiere probar con la mezcla antes de armarla.

Uso sugerido de la mezcla: los **digitales** prueban escala (muchos drones, batching, dedupe en la grilla) y los
**fisicos** prueban la cadena real (video real, pose real, georef real, detector en imagenes reales).

## Preguntas para integracion

- [x] Prueba: 2 drones fisicos + drones digitales (Gazebo).
- [ ] Cuantos drones digitales, y si comparten mundo/grilla con los fisicos.
- [ ] Cuantos drones simultaneos por outpost en operacion, y cuantos outposts.
- [ ] Protocolo y codec del video (RTSP/UDP, H.264/H.265), resolucion y fps de cada flujo.
- [ ] ¿Drones mixtos (algunos RGB, otros IR) o todos cambian juntos dia/noche?
- [ ] ¿La pose de cada dron llega sincronizada con su video (timestamps comunes)?

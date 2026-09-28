# Plan: validación en simulación (Gazebo) del módulo de georreferenciación

## Contexto
- Mi módulo: detección IA (YOLO) + georreferenciación por raycasting sobre DEM (Rossi et al. 2025), ya empaquetado como nodo ROS 2.
- Stack heredado: contenedor Docker con ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL + YOLOv8. Mundo actual: "rubicon world" (aún no revisado).
- Hardware local: laptop RTX 3050, 4 GB VRAM, Kubuntu 24.04. Sin acceso al PC del laboratorio (está en otra región).
- Plazo: ~1 mes para validar mi parte, integrar mis módulos y luego integrar con el orquestador y la programación de drones.
- Requisito: validar con VARIOS drones. Prioridad: mínima fricción de entorno.

## Decisiones tomadas
1. Cómputo
   - Primero, prueba local headless: 2 drones, cámaras a 640×480, YOLOv8n FP16, Gazebo con `-s --headless-rendering`. Medir con `nvidia-smi`.
   - Si supera ~3.5 GB de VRAM o se necesitan 3 o más drones: VM con GPU alquilada (16-24 GB), con la MISMA imagen Docker (push a GHCR, pull en la VM). `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`.
   - Sin escritorio remoto: Foxglove Bridge por túnel SSH para visualizar. Correr escenarios en lote, grabar rosbags, descargarlos y apagar la VM.
2. Validación por capas
   - Georref offline con rosbags (imagen + telemetría + ground truth) en CPU.
   - Detección por separado.
   - Lazo cerrado solo en sesiones cortas.
3. Mundo 3D
   - Terreno real a partir de un DEM (Copernicus GLO-30) como heightmap. NO usar Google Earth 3D Tiles (licencia, peso).
   - `<spherical_coordinates>` en el SDF = origen geográfico del mundo.
4. Una sola fuente de verdad
   - El mismo DEM genera el heightmap de Gazebo y el DEM del raycasting.
   - Las posiciones de objetos (lat/lon) generan tanto el mundo como el ground truth.
5. Colocación de objetos
   - Manual en la GUI de Gazebo (Resource Spawner + transform tool, sim pausada, sin drones, `<static>true</static>`, "Save world as" en un archivo nuevo), o bien pines en Google Earth Pro/QGIS → CSV → script.
   - En ambos casos, un script extrae el ground truth del SDF.
6. Detector oráculo
   - Proyectar la posición real de cada persona a la cámara y generar un bbox con ruido controlado.
   - Valida la georref independiente de YOLO. YOLO real queda solo como demo de la cadena completa.
7. Validación en dos fases
   - (a) Mismo DEM, sin ruido: error ≈ 0 (verifica la implementación).
   - (b) DEM degradado en el raycast + ruido de telemetría: error realista (resultado de la tesis). Métricas: media, RMSE y P95 por escenario.

## Pasos con checkpoints
- Día 1: descargar el DEM, reproyectar y generar el heightmap:
  `gdalwarp -t_srs "+proj=tmerc +lat_0=LAT +lon_0=LON +k=1 +datum=WGS84" -te -2000 -2000 2000 2000 -ts 1025 1025 -r bilinear dem.tif local.tif`
  `gdal_translate -ot UInt16 -scale MIN MAX 0 65535 -of PNG local.tif terreno.png`
  Heightmap: `<size>4000 4000 (MAX-MIN)</size>`, `<pos>0 0 (MIN-ELEV_ORIGEN)</pos>`, tanto en visual como en collision.
  CHECKPOINT: volar sobre un cerro conocido; el GPS de PX4 debe coincidir con lat/lon/alt del DEM.
- Día 2: script `escenario.csv` → `<include>` en SDF (lat/lon → ENU con pymap3d, Z desde el DEM con rasterio) y script `SDF` → `ground_truth.csv`. Outposts como puntos de spawn de los drones.
- Día 3: multi-dron + detector oráculo + error de punta a punta.
- Luego: escenarios en lote + rosbags → análisis offline (fases a y b).

## Trampas conocidas
- UTM rota el mundo (~0.1° de convergencia en Colchane): usar tmerc centrado en el origen.
- Curvatura: mundo de 3-5 km de lado como máximo (~2 m de error vertical a 5 km).
- PNG de 8 bits produce un terreno escalonado: usar 16 bits.
- Heightmap posiblemente espejado: lo detecta el checkpoint del día 1.
- Guardar desde la GUI reescribe el SDF: guardar en un archivo nuevo, sin drones cargados.

## Pendiente
- Definir la zona exacta (lat/lon del centro).
- Revisar el SDF de rubicon world: `<heightmap>`/`<mesh>`, `<spherical_coordinates>`, `<include>`.
- Resultado de la prueba local de VRAM (decide si hace falta la VM o no).
- Textura satelital (Sentinel-2): opcional, al final.
- Scripts a escribir: `generar_mundo.py`, `extraer_gt.py`, detector oráculo, lanzador de escenarios en lote, análisis de error.
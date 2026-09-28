---
title: "Capa de analisis: de detecciones georreferenciadas a alertas para el orquestador"
fecha: 2026-09-28
estado: diseno (sin implementar)
---

# Capa de analisis y alertas

## Problema

El detector genera muchas detecciones por segundo, ruidosas, con IDs de tracking inestables.
El orquestador de drones necesita **pocas alertas, con georreferencia, de un solo tipo**:
"hay presencia humana persistente aqui". No necesita identidad individual.

Entre ambos falta una capa que **acumule evidencia en el mapa y decida cuando alertar**.

## Pipeline

```
Detector (por frame)          bbox + conf + clase, RGB o IR
  -> Filtro de deteccion      descartar lo que no sirve (ver 1)
  -> Georreferencia           ray casting al DEM -> celda de 12.5 m + incertidumbre
  -> Grilla de evidencia      acumula en el tiempo, olvida lo viejo (ver 2)
  -> Zonas                    agrupa celdas calientes vecinas (ver 3)
  -> Maquina de estados       decide nueva / actualiza / cerrada (ver 4)
  -> Publicador               rate limit + deduplicacion -> orquestador (ver 5)
```

El filtro que pedia el orquestador no es uno solo: son **tres niveles** (deteccion, tiempo, evento).

## 1. Filtro por deteccion (antes de georreferenciar)

- Solo clase `person` (contrato acotado).
- `conf` minima.
- Punto a proyectar: **centro inferior de la caja** (los pies), no el centro.
- Descartar rayos degenerados: casi horizontales, que no intersectan el DEM, o que caen a mas de X m del dron
  (a mayor distancia, mayor error de georreferencia).
- Guardar la incertidumbre de posicion: crece con la distancia y lo oblicuo del rayo.
  Error de actitud de 1 grado a 100 m de distancia inclinada ~ 1.7 m; mas lejos y mas oblicuo, bastante mas.

## 2. Grilla de evidencia (sobre las celdas del DEM, 12.5 m)

Estado por celda:

| Campo | Para que |
|-------|----------|
| `L` (log-odds de presencia) | evidencia acumulada, acotada a [L_min, L_max] |
| `n_obs`, `n_hits` (ventana deslizante) | hit ratio = en que % de las veces que se miro la celda hubo deteccion |
| `first_seen`, `last_seen` | persistencia en segundos |
| `conf_max`, `sensor` | que tan fuerte y de donde (rgb/ir) |

Actualizacion por frame:

- **Celda observada y con deteccion:** `L += w_hit * conf` (repartido a vecinas si la incertidumbre es mayor que la celda).
- **Celda observada y sin deteccion:** `L -= w_miss` (evidencia negativa: se miro y no habia nada).
- **Celda NO observada** (fuera del footprint de la camara): **no se toca**, o decae muy lento.
  Critico: si el dron gira, lo que sale de cuadro no debe "enfriarse" como si estuviera vacio.
- Decaimiento temporal: `L *= lambda`, con `lambda` derivado de una vida media en segundos:
  `lambda = 0.5 ** (1 / (fps * t_media))`. Parametros con sentido fisico, faciles de explicar y ajustar.

Por que log-odds: es la formulacion estandar de occupancy grids; evidencia positiva y negativa simetricas, y el
acotamiento hace que una celda se recupere rapido cuando la situacion cambia.

Necesita **footprint de la camara por frame** (poligono en el suelo visto por la camara): sale del mismo
ray casting con las 4 esquinas de la imagen.

## 3. Zonas

- Componentes conexas (vecindad 8) de celdas con `L > T_on`.
- Una zona = candidata a "presencia humana". Centroide ponderado por `L`.
- Seguimiento de zonas, no de personas, a baja frecuencia (ej. 1 Hz): asociar por vecino mas cercano con
  compuerta de velocidad (persona a pie <= ~2 m/s: menos de 1 celda por segundo).
  Es mucho mas estable que el tracking por caja porque la grilla ya suavizo el ruido.
- Secuencia de centroides -> **traza**: velocidad y rumbo de la zona.

## 4. Maquina de estados por zona

```
CANDIDATA --(persiste >= t_confirm y hit_ratio >= r_min)--> CONFIRMADA --> emite ALERTA "nueva"
CONFIRMADA --(se movio > d o cambio conf, y paso >= t_min_update)--> emite "actualiza"
CONFIRMADA --(L < T_off durante t_perdida)--> CERRADA --> emite "cerrada"
```

- **Histeresis**: `T_off < T_on`. Evita alertas que parpadean al borde del umbral.
- `t_confirm` es el costo en latencia: cuantos segundos de persistencia antes de molestar al orquestador.

## 5. Contrato con el orquestador (borrador)

```json
{
  "alert_id": "uuid",
  "type": "human_presence",
  "state": "new | update | closed",
  "lat": 0.0, "lon": 0.0, "alt_dem": 0.0,
  "radius_m": 18.0,
  "confidence": 0.82,
  "first_seen": "2026-09-28T04:10:00Z",
  "last_seen": "2026-09-28T04:10:42Z",
  "velocity": {"speed_mps": 1.1, "heading_deg": 240},
  "sensor": "rgb | ir",
  "n_detections": 57,
  "source_drone": "uav-1"
}
```

Reglas de envio:
- `new`: inmediato al confirmar.
- `update`: como maximo 1 cada N s, y solo si cambio algo relevante (posicion > d, confianza, estado de movimiento).
- `closed`: al perder la zona.
- Nunca se envian detecciones crudas.

## 6. Como desarrollarlo sin volar: log + replay

Lo mas util que se puede hacer **ya**: definir un log de detecciones y reprocesarlo offline.

Por frame: `timestamp, drone_id, pose (lat, lon, alt, roll, pitch, yaw), gimbal, sensor, [bbox, conf, clase]*`.

Con eso, la capa de analisis se ajusta en replay (umbrales, vida media, t_confirm) sin gastar vuelos,
y los mismos datos sirven para comparar parametros. Requiere **sincronizar tiempo del video con telemetria**.

## 7. Como evaluar esta capa (no es mAP)

- Tiempo hasta la alerta (desde que la persona aparece).
- Falsas alertas por hora.
- Error de posicion de la alerta (m) contra la posicion real.
- Personas persistentes nunca alertadas.
- Estabilidad: cuantas alertas `new` genera una sola persona (ideal: 1).

Las simulaciones con personas son el banco de prueba natural: posicion real conocida.

## 8. Relacion con el resto

- Separado del framework de entrenamiento (`src/deteccion`): es otro nodo, consume detecciones.
- Destino: pipeline ROS2 + YOLO + Gazebo (base heredada) en Linux. Nodos probables:
  `detector` (YOLO) -> `georef` (ray casting al DEM) -> `alertas` (grilla + zonas + estados) -> orquestador.
  El log de la seccion 6 equivale a un `ros2 bag` de los topics de detecciones + pose: replay nativo de ROS2.
- Gazebo da posiciones reales de las personas simuladas: ground truth para las metricas de la seccion 7.
- RGB (dia) e IR (noche) escriben en **la misma grilla**: el toggle de sensor no cambia nada aguas abajo.
  Si algun dia hay varios drones, tambien escriben en la misma grilla.
- El tracker queda como apoyo (tracklets cortos dan velocidad), no como fuente de verdad.

## Preguntas abiertas

- [x] Ray casting en tiempo real: **Python sobre ROS2**. La capa de analisis es un nodo ROS2 mas.
- [ ] Formato hacia el orquestador: topic ROS2 con mensaje propio (campos de la seccion 5) vs JSON en un `std_msgs/String`.
- [ ] Frecuencia que espera el orquestador.
- [ ] Latencia maxima aceptable para una alerta (define `t_confirm`).
- [ ] Precision de la pose del dron (IMU/GPS/gimbal): define la incertidumbre y si 12.5 m alcanza.
- [ ] Peso relativo de RGB vs IR en la grilla (IR de noche, ¿mas o menos confiable?).
- [ ] ¿Alertas de vehiculos en el futuro? Misma grilla con capa por clase.

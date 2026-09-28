# Notas de investigacion

| Nota | Tema | Estado |
|------|------|--------|
| [plan_refactor.md](plan_refactor.md) | Por que y como se ordeno el framework de entrenamiento (razonamiento detras de `docs/SPEC.md`) | hecho (F0-F5) |
| [capa_alertas.md](capa_alertas.md) | De detecciones georreferenciadas a alertas para el orquestador: grilla de evidencia, zonas, estados | diseno |
| [pipeline_colab.md](pipeline_colab.md) | Investigacion inicial: como sincronizar codigo con Colab (antes del refactor) | historico |

Spec implementable del framework: [`../docs/SPEC.md`](../docs/SPEC.md).

## Siguiente etapa

Portar a Linux sobre el pipeline base ROS2 + YOLO + Gazebo: entorno de simulacion propio, nodos de georreferencia y alertas.

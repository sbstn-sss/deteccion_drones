"""Framework para entrenar y evaluar detectores YOLO (VisDrone, HIT-UAV, ...)."""

from deteccion import data, env, viz

__all__ = ["data", "env", "viz", "ExperimentConfig", "Experiment", "list_runs"]


def __getattr__(name: str):
    # experiment.py importa ultralytics; perezoso para que data/env/viz y los tests
    # corran en local sin torch instalado.
    if name in ("ExperimentConfig", "Experiment", "list_runs"):
        from deteccion import experiment

        return getattr(experiment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

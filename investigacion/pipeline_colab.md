---
title: "Investigacion: pipeline Colab + YOLO con codigo modular y experiment tracking"
fecha: 2026-06-24
estado: investigacion
---

# Pipeline Colab + YOLO: investigacion previa

## Problema actual
- Notebook `hola.ipynb` con todo el codigo pegado en celdas (setup, train, eval, tracking).
- Cada sesion de Colab es efimera: se borra `/content`, hay que remontar drive, descomprimir dataset, reinstalar deps.
- No hay seguimiento de modelos entrenados (registro manual en `proceso.md`).
- Se quiere: codigo en `.py` aparte, 1 comando de inicializacion, tracking automatico.

---

## 1. Sincronizacion carpeta local -> Colab

El kernel de Colab corre en la VM de Google (no en tu Windows). Tus `.py` locales NO son visibles directamente. Hay que llevarlos a la VM. Opciones ordenadas por facilidad:

### Opcion A: Google Drive para Desktop (LA MAS FACIL) ⭐
- Instalas la app oficial **Drive para Escritorio** de Google.
- Drive se monta como unidad en Windows (ej. `G:\Mi unidad\`).
- Editas tus `.py` con VS Code directamente en `G:\Mi unidad\deteccion\src\`.
- En Colab, `drive.mount('/content/drive')` los ve **inmediatamente**. Cero comandos de upload.
- Sincronizacion automatica bidireccional.
- **Desventaja**: I/O de Drive es lento para muchos archivos chicos. El dataset grande (10k+ imagenes) NO debe leerse directo desde Drive montado. Mejor dejarlo como `.zip` en Drive y descomprimir a `/content` al inicio (lo que ya haces).
- **Conclusión**: ideal para el **codigo** (`src/`, configs), NO para el dataset crudo.

### Opcion B: GitHub (LA MAS PROFESIONAL)
- Repo local con git, push a GitHub desde VS Code.
- En Colab: `!git clone https://github.com/USER/deteccion.git /content/deteccion` (1 comando).
- Versionado real, reproducibilidad, ramas para experimentos.
- **Desventaja**: requiere cuenta GitHub + aprender git basico. El dataset NO va en git (demasiado grande, se ignora con `.gitignore`).
- **Conclusión**: ideal si quieres versionado serio del codigo. Se combina con Drive para el dataset.

### Opcion C: rclone (CLI avanzado)
- `rclone copy C:\...\src drive:deteccion/src` sincroniza por linea de comandos.
- Bidireccional, potente, scriptable.
- **Desventaja**: setup inicial complejo (configurar remote de Drive con OAuth).
- **Conclusión**: util si quieres un script `sync.ps1` local, pero overkill para empezar.

### Recomendacion
**Empezar con Opcion A (Drive para Desktop)** porque ya usas Drive y es cero friccion. Si despues quieres versionado, migrar a Opcion B (GitHub) es directo. Ambas pueden coexistir: codigo en GitHub + dataset/modelos en Drive.

---

## 2. Persistencia entre sesiones efimeras

Cada vez que te conectas a Colab, `/content` esta vacio. Hay que automatizar:

| Que | Donde persiste | Estrategia |
|-----|---------------|-----------|
| Codigo `src/` | Drive (o GitHub) | Montar drive + agregar a `sys.path` |
| Dataset | Drive como `.zip` | `!unzip` a `/content/dataset` al inicio |
| Pesos `.pt` | Drive | Se guardan ahi con `project=` en `model.train()` |
| W&B logs | Nube W&B | Automatico, no se pierde |
| Configs YAML | Drive | Se versionan con el codigo |

**Flujo ideal de `init_colab()`** (1 sola funcion que lo hace todo):
1. Montar Drive.
2. Agregar `src/` al `sys.path`.
3. `pip install ultralytics wandb` (silencioso si ya estan).
4. Si `/content/dataset` no existe: `unzip` desde Drive.
5. Fix del `path:` en `visdrone.yaml`.
6. Login W&B.
7. Verificar GPU disponible.

Tiempo de setup: ~2-3 min por sesion (dominado por el unzip). Solo se corre 1 vez al iniciar.

---

## 3. Experiment tracking (registro de modelos)

Ultralytics tiene integraciones nativas. Comparativa:

| Tool | Setup | Gratis | Dashboard | Comparar runs | Recomendado |
|------|-------|--------|-----------|--------------|-------------|
| **W&B (Weights & Biases)** | 3 lineas | Si (personal) | Excelente, web | Si, visual | **SI** ⭐ |
| TensorBoard | Ya viene | Si | Basico, local | Manual | Backup |
| MLflow | Medio | Si (self-host) | Bueno | Si | Overkill |
| ClearML / Comet / Neptune | Medio | Limitado | Bueno | Si | Alternativas |

### W&B con Ultralytics (integracion nativa confirmada en docs)
```python
pip install -U ultralytics wandb
yolo settings wandb=True          # activa logging automatico
import wandb; wandb.login(key="API_KEY")  # 1 vez
```
Despues, cada `model.train()` loguea automaticamente a W&B:
- Metricas (loss, mAP, precision, recall) en tiempo real.
- Hiperparametros (epochs, imgsz, batch, lr).
- Artefactos (pesos del modelo).
- Comparacion visual entre runs (ej. `experimento_1` vs `experimento_1_50_epochs`).
- Dashboard web en `wandb.ai` — accesible desde cualquier dispositivo.

**API key**: se obtiene en https://wandb.ai/authorize (cuenta google). Gratis para uso personal (hasta cierto limite de storage).

### Que resuelve para ti
Hoy llevas registro manual en `proceso.md` ("experimento_1_50_epochs"). Con W&B:
- Cada run queda registrado automaticamente con todos sus hiperparametros.
- Puedes comparar curvas de loss/mAP entre runs side-by-side.
- No se pierde entre sesiones de Colab (vive en la nube W&B).
- Tags y notas para organizar (ej. tag "visdrone", "yolov8s").

---

## 4. Arquitectura propuesta del proyecto

```
deteccion/
├── hola.ipynb              # notebook slim: solo importa src y ejecuta
├── src/
│   ├── __init__.py
│   ├── setup.py            # init_colab(): drive, deps, unzip, yaml, wandb, GPU check
│   ├── data.py             # mapeo pandas + visualizacion de muestras
│   ├── train.py            # run_training(config): wrapper de model.train()
│   ├── evaluate.py         # run_eval(): comparativa prediccion vs ground truth
│   └── tracking.py         # run_tracking(): video + BotSort
├── configs/
│   ├── base.yaml           # defaults (imgsz=1024, batch=16, patience=5)
│   └── experimentos/
│       ├── exp1_15ep.yaml  # overrides
│       └── exp1_50ep.yaml
├── investigacion/          # docs (ya existe)
└── requirements.txt        # deps pinneadas para reproducibilidad
```

### Notebook slim resultante (~3 celdas)
```python
# Celda 1: setup (corre 1 vez por sesion)
from src.setup import init_colab
init_colab()  # drive, deps, unzip, wandb, GPU

# Celda 2: entrenar
from src.train import run_training
run_training(config="configs/experimentos/exp1_50ep.yaml")

# Celda 3: evaluar + tracking
from src.evaluate import run_eval
from src.tracking import run_tracking
run_eval(model_name="experimento_1_50_epochs")
run_tracking(video="video1_testing.mp4")
```

---

## 5. Lo que necesitas instalar / configurar (checklist)

### Minimo para empezar
- [ ] **Google Drive para Desktop** (app): https://www.google.com/drive/download/ — opcional pero recomendado para editar `.py` comodo en VS Code.
- [ ] **Cuenta W&B**: gratis en https://wandb.ai — obtener API key de https://wandb.ai/authorize.
- [ ] **Python local** (ya tienes 3.13) — solo para editar, no para entrenar.

### Opcional (despues)
- [ ] **GitHub**: para versionado del codigo. Si quieres, despues migramos.
- [ ] **Colab Pro**: sesiones mas largas y GPU mejor. No es obligatorio para empezar.

---

## 6. Riesgos / consideraciones

1. **I/O de Drive**: NO entrenar leyendo imagenes directo desde `/content/drive/...`. Siempre descomprimir a `/content/dataset` primero (Drive montado es 10x mas lento que FS local de la VM).
2. **Sesiones efimeras**: Colab free desconecta por inactividad (~90 min) y limite de horas (~12h). Con `patience=5` y checkpoints en Drive, esto se mitiga.
3. **W&B gratis**: limite de storage (100GB en plan free). Los pesos de YOLOv8s son ~22MB c/u, cabe holado. Si entrenas MUCHO, revisar.
4. **Drive para Desktop**: ocupa espacio local si usas "Espejo" (Mirror). Usar modo "Transmitir" (Streaming) para no llenar el disco. La app te pregunta al instalar.
5. **Compatibilidad VS Code + Colab**: la extension de Colab para VS Code solo cambia donde se ejecuta el kernel. El filesystem sigue siendo el de la VM. Tus `.py` locales no son visibles salvo que esten en Drive.

---

## Siguientes pasos propuestos

1. **Decidir**: Opcion A (Drive para Desktop) vs Opcion B (GitHub) para el codigo.
2. **Crear cuenta W&B** y guardar API key.
3. **Implementar** `src/` con los 5 modulos + `init_colab()`.
4. **Slim el notebook** `hola.ipynb` a ~3 celdas.
5. **Probar** el pipeline end-to-end en una sesion de Colab.
```

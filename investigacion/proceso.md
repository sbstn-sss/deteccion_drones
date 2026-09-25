---
title: "Investigacion para hacer el primer detector"
---
### Consideraciones Colab
1. Todo setting hay que hacerlo con CPU gratis. 
2. Solo cambiar a GPU dedicada al momento de ENTRENAR los modelos
3. Early stopping para no quemar creditos

### En que estoy
- Estoy descargando el dataset VisDrone2019 y quiero importarlo a drive para poder empezar a trabajar desde el notebook
- Entrenando el primer modelo (Luego tengo que tener una rutina para probarlo jeje)
  - El primer modelo se llama: "primer_training-2"
  - El modelo entrenado con 1024 de img size fue "experimento_1" 
  - Dejare corriendo uno por 50 epocas ("experimento_1_50_epochs")
  - Estoy entrenando un modelo yolo11s con 1024 de img size y 8 de batch. Mientras cargo el MOT dataset y lo preparo
  ... En proceso
  
Despues:
  - Luego de cargar el dataset y entrenar un primer modelo rapidamente y probarlo, haremos lo siguiente. Hare una investigacion en un obsidian nuevo e ire documentando todo lo que halle. 
    - Esta investigacion debe estar centrada en el siguiente hecho. Tengo que aprender a armar un sistema que haga lo siguiente:
      - Deteccion RGB -> dataset a usar: VisDrone
      - Deteccion termica -> debo investigar como hacerla
      - Multispectral sensing -> Fusion de detectores -> como genero una alerta de una amenaza?
      - Georreferencia desde imagen. Que datos necesito, como puedo probarlo con un dataset real con data geoespacial de la camara y con datos geospaciales reales. Esto me tinca que lo hare en matlab

- Decidi que deteccion RGB es lo inicial, deteccion termica la vere despues. Problema de video. 
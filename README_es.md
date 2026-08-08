# GD RL Agent

Agente de aprendizaje por refuerzo que aprende a jugar Geometry Dash (Steam) mediante captura de pantalla e inyección de teclado. Proyecto educativo — sin frameworks de RL, todo implementado desde cero con PyTorch.

## Restricciones clave

- Solo CPU (Intel i7-1225U, sin GPU)
- Entorno en tiempo real — sin aceleración de simulación ni paralelización
- Espacio de acciones binario: saltar / no saltar (solo modo cubo)
- Entrenamiento en Stereo Madness 0–29% (segmento cubo, sin portales)

## Inicio rápido

```bash
pip install -e .
python cli.py calibrate        # buscar ventana, medir latencia, guardar config
python cli.py debug            # verificar visualmente el pipeline de visión
python cli.py train --episodes 500 --run-name vanilla
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt
```

## Variantes DQN

```bash
python cli.py train --no-double --no-dueling --run-name vanilla
python cli.py train --double    --no-dueling --run-name double
python cli.py train --no-double --dueling    --run-name dueling
```

## Arquitectura

```
gd-rl-agent/
├── capture/       # captura de pantalla mss, detección de ventana
├── input/         # inyección de salto pydirectinput, medición de latencia
├── vision/        # detección HSV de obstáculos/hoyos/muerte, extractor de estado
├── env/           # bucle de entorno: reset() / step() / close()
├── agent/         # QNetwork, ReplayBuffer, Trainer
├── configs/       # modelos de configuración Pydantic v2
├── cli.py         # CLI Typer: calibrate / debug / train / watch / eval
└── tests/         # tests unitarios de agent/ (sin juego real)
```

## Fórmula de recompensa

```
r_t = +0.01   por cada paso sobrevivido
    + (-1.0)  al morir
    + (+5.0)  al completar el segmento
    + (-0.02) por salto inútil (en suelo, sin obstáculo en el radio umbral)
γ = 0.99
```

## Vector de características (4 floats)

| Característica | Rango | Centinela |
|---|---|---|
| `dist_obstacle` | [0, 1] | 1.0 |
| `dist_pit` | [0, 1] | 1.0 |
| `obstacle_height` | [0, 1] | 0.0 |
| `is_grounded` | {0, 1} | — |

## Tests

```bash
python -m pytest tests/ -v
```

Guía completa de instalación y calibración en `INSTRUCTION_es.md`.

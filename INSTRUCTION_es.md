# GD RL Agent — Guía de instalación y uso

Un agente de aprendizaje por refuerzo que aprende a jugar Geometry Dash (Steam) mediante captura de pantalla e inyección de teclado. Sin frameworks de RL — todo implementado desde cero con PyTorch.

## Requisitos

- Windows 10/11 (pydirectinput requiere Windows)
- Python 3.12 o superior
- Geometry Dash (Steam, versión 2.2)
- No se requiere GPU — funciona solo con CPU

## Instalación

```bash
git clone <repo-url>
cd gd-rl-agent
pip install -e .
```

## Paso 1 — Calibración

Inicia Geometry Dash en **modo ventana** y abre Stereo Madness en **modo práctica (practice mode)**.

```bash
python cli.py calibrate
```

Este comando:
1. Encuentra la ventana de GD automáticamente
2. Mide los FPS de captura y la latencia de entrada
3. Toma una captura de pantalla — abre `calibration_frame.png` e introduce las coordenadas de píxeles cuando se soliciten
4. Guarda la configuración en `configs/agent_config.json`

**Valores a introducir:**
- `player_right_x` — píxel X en el borde derecho del sprite del cubo
- `ground_y` — píxel Y en el borde superior de las baldosas del suelo
- `cube_height_px` — altura del cubo en píxeles (medir en la captura)

## Paso 2 — Verificar el pipeline de visión

```bash
python cli.py debug --duration 10
```

Observa los vectores de características impresos y comprueba `debug_frame.png`. El rectángulo verde es el ROI de escaneo, la línea vertical roja marca el obstáculo detectado y la naranja marca un hoyo. Si la detección no funciona correctamente, hay que ajustar empíricamente los umbrales HSV en `vision/obstacle_detector.py`.

## Paso 3 — Entrenamiento

Activa **Auto-Retry** en los ajustes de GD. Inicia Stereo Madness en modo práctica y coloca un checkpoint al principio del nivel (tecla `Z`). Luego ejecuta:

```bash
# DQN estándar
python cli.py train --episodes 500 --run-name vanilla

# Double DQN
python cli.py train --episodes 500 --double --run-name double

# Dueling DQN
python cli.py train --episodes 500 --dueling --run-name dueling
```

Los registros de entrenamiento se guardan en `runs/<run-name>/train_log.csv`. Los checkpoints se guardan cada 50 episodios.

**Importante:** Tras la primera ejecución, comprueba los FPS reales en el registro y recalibra `max_episode_steps` en `configs/agent_config.json` si difieren significativamente del valor de calibración.

## Paso 4 — Observar y evaluar

```bash
# Observar al agente jugar (sin actualizar pesos)
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 5

# Evaluar recompensa media y tasa de éxito
python cli.py eval --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 20
```

## Referencia de configuración

Todos los ajustes se encuentran en `configs/agent_config.json` (creado por `calibrate`).

| Campo | Valor por defecto | Descripción |
|---|---|---|
| `episode.max_episode_steps` | 250 | Pasos por episodio — calibrar con DQN en marcha |
| `episode.auto_retry` | true | true si Auto-Retry de GD está activado |
| `reward.r_step` | 0.01 | Recompensa por cada paso sobrevivido |
| `reward.r_death` | -1.0 | Penalización por muerte |
| `reward.r_completion` | 5.0 | Bonificación al alcanzar el final del segmento |
| `reward.penalize_useless_jumps` | true | Penalizar saltos sin obstáculo por delante |
| `reward.useless_jump_threshold` | 0.35 | Umbral de distancia para salto "inútil" |
| `hp.use_double_dqn` | false | Activar Double DQN |
| `hp.use_dueling_dqn` | false | Activar Dueling DQN |
| `hp.use_obstacle_lookahead` | false | Añadir segundo obstáculo al vector de características |

## Estructura del proyecto

```
gd-rl-agent/
├── capture/          # Captura de pantalla mss, detección de ventana
├── input/            # Inyección de salto via pydirectinput
├── vision/           # Detección HSV de obstáculos/hoyos/muerte, extractor de estado
├── env/              # Bucle de entorno personalizado (reset/step/close)
├── agent/            # QNetwork, ReplayBuffer, Trainer
├── configs/          # Modelos de configuración Pydantic
├── cli.py            # CLI Typer (calibrate/debug/train/watch/eval)
└── tests/            # Tests unitarios de agent/ (sin juego real)
```

## Ejecutar tests

```bash
python -m pytest tests/ -v
```

# GD RL Agent

Ein Reinforcement-Learning-Agent, der Geometry Dash (Steam) durch Bildschirmaufnahme und Tastatureingabe-Injektion erlernt. Bildungsprojekt — keine RL-Frameworks, alles von Grund auf mit PyTorch implementiert.

## Rahmenbedingungen

- Nur CPU (Intel i7-1225U, keine GPU erforderlich)
- Echtzeit-Umgebung — keine Simulationsbeschleunigung, keine Parallelisierung
- Binärer Aktionsraum: Springen / Nicht springen (nur Würfelmodus)
- Training auf Stereo Madness 0–29% (Würfelsegment, keine Portale)

## Schnellstart

```bash
pip install -e .
python cli.py calibrate        # Fenster finden, Latenz messen, Konfig speichern
python cli.py debug            # Vision-Pipeline visuell überprüfen
python cli.py train --episodes 500 --run-name vanilla
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt
```

## DQN-Varianten

```bash
python cli.py train --no-double --no-dueling --run-name vanilla
python cli.py train --double    --no-dueling --run-name double
python cli.py train --no-double --dueling    --run-name dueling
```

## Architektur

```
gd-rl-agent/
├── capture/       # mss-Bildschirmaufnahme, Fenstererkennung
├── input/         # pydirectinput Sprung-Injektion, Latenzmessung
├── vision/        # HSV-Erkennung von Hindernissen/Gruben/Tod, State extractor
├── env/           # Umgebungsschleife: reset() / step() / close()
├── agent/         # QNetwork, ReplayBuffer, Trainer
├── configs/       # Pydantic v2 Konfigurationsmodelle
├── cli.py         # Typer CLI: calibrate / debug / train / watch / eval
└── tests/         # Unit-Tests für agent/ (ohne echtes Spiel)
```

## Belohnungsformel

```
r_t = +0.01   pro überlebtem Schritt
    + (-1.0)  beim Tod
    + (+5.0)  bei Segmentabschluss
    + (-0.02) bei nutzlosem Sprung (geerdet, kein Hindernis im Schwellenwertbereich)
γ = 0.99
```

## Feature-Vektor (4 Floats)

| Feature | Bereich | Sentinel |
|---|---|---|
| `dist_obstacle` | [0, 1] | 1.0 |
| `dist_pit` | [0, 1] | 1.0 |
| `obstacle_height` | [0, 1] | 0.0 |
| `is_grounded` | {0, 1} | — |

## Tests

```bash
python -m pytest tests/ -v
```

Vollständige Installations- und Kalibrierungsanleitung in `INSTRUCTION_de.md`.

# GD RL Agent — Installations- und Nutzungsanleitung

Ein Reinforcement-Learning-Agent, der Geometry Dash (Steam) durch Bildschirmaufnahme und Tastatureingabe-Injektion erlernt. Keine RL-Frameworks — alles von Grund auf mit PyTorch geschrieben.

## Voraussetzungen

- Windows 10/11 (pydirectinput erfordert Windows)
- Python 3.12 oder neuer
- Geometry Dash (Steam-Version 2.2)
- Keine GPU erforderlich — läuft auf CPU

## Installation

```bash
git clone <repo-url>
cd gd-rl-agent
pip install -e .
```

## Schritt 1 — Kalibrierung

Starte Geometry Dash im **Fenstermodus** und öffne Stereo Madness im **Übungsmodus (practice mode)**.

```bash
python cli.py calibrate
```

Dieser Befehl:
1. Findet das GD-Fenster automatisch
2. Misst die Aufnahme-FPS und Eingabelatenz
3. Erstellt einen Screenshot — öffne `calibration_frame.png` und gib die Pixelkoordinaten ein
4. Speichert die Konfiguration in `configs/agent_config.json`

**Einzugebende Werte:**
- `player_right_x` — X-Pixel am rechten Rand des Würfel-Sprites
- `ground_y` — Y-Pixel am oberen Rand der Bodenkacheln
- `cube_height_px` — Höhe des Würfels in Pixeln (im Screenshot messen)

## Schritt 2 — Vision-Pipeline prüfen

```bash
python cli.py debug --duration 10
```

Beobachte die ausgegebenen Feature-Vektoren und prüfe `debug_frame.png`. Das grüne Rechteck ist der Scan-ROI, die rote Linie markiert ein erkanntes Hindernis, orange markiert eine Grube. Falls die Erkennung nicht funktioniert, müssen die HSV-Schwellwerte in `vision/obstacle_detector.py` empirisch angepasst werden.

## Schritt 3 — Training

Aktiviere **Auto-Retry** in den GD-Einstellungen. Starte Stereo Madness im Übungsmodus und setze einen Checkpoint ganz am Anfang des Levels (Taste `Z`). Dann:

```bash
# Vanilla DQN
python cli.py train --episodes 500 --run-name vanilla

# Double DQN
python cli.py train --episodes 500 --double --run-name double

# Dueling DQN
python cli.py train --episodes 500 --dueling --run-name dueling
```

Trainingslogs werden in `runs/<run-name>/train_log.csv` gespeichert. Checkpoints werden alle 50 Episoden gespeichert.

**Wichtig:** Nach dem ersten Training die tatsächliche FPS im Log prüfen und `max_episode_steps` in `configs/agent_config.json` neu kalibrieren, falls sie deutlich vom Kalibrierungswert abweicht.

## Schritt 4 — Beobachten und Auswerten

```bash
# Agenten beim Spielen beobachten (ohne Gewichtsaktualisierung)
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 5

# Mittlere Belohnung und Erfolgsrate auswerten
python cli.py eval --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 20
```

## Konfigurationsreferenz

Alle Einstellungen befinden sich in `configs/agent_config.json` (erstellt durch `calibrate`).

| Feld | Standard | Beschreibung |
|---|---|---|
| `episode.max_episode_steps` | 250 | Schritte pro Episode — mit laufendem DQN kalibrieren |
| `episode.auto_retry` | true | true, wenn GD Auto-Retry aktiviert ist |
| `reward.r_step` | 0.01 | Belohnung pro überlebtem Schritt |
| `reward.r_death` | -1.0 | Strafe beim Tod |
| `reward.r_completion` | 5.0 | Bonus beim Erreichen des Episodenendes |
| `reward.penalize_useless_jumps` | true | Sprünge ohne Hindernis bestrafen |
| `reward.useless_jump_threshold` | 0.35 | Distanzschwelle für „nutzlosen" Sprung |
| `hp.use_double_dqn` | false | Double DQN aktivieren |
| `hp.use_dueling_dqn` | false | Dueling DQN aktivieren |
| `hp.use_obstacle_lookahead` | false | Zweites Hindernis zum Feature-Vektor hinzufügen |

## Projektstruktur

```
gd-rl-agent/
├── capture/          # mss-Bildschirmaufnahme, Fenstererkennung
├── input/            # Sprung-Injektion via pydirectinput
├── vision/           # HSV-Erkennung von Hindernissen/Gruben/Tod, State extractor
├── env/              # Umgebungsschleife (reset/step/close)
├── agent/            # QNetwork, ReplayBuffer, Trainer
├── configs/          # Pydantic-Konfigurationsmodelle
├── cli.py            # Typer CLI (calibrate/debug/train/watch/eval)
└── tests/            # Unit-Tests für agent/ (ohne echtes Spiel)
```

## Tests ausführen

```bash
python -m pytest tests/ -v
```

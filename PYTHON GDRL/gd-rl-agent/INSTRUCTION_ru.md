# GD RL Agent — Руководство по установке и использованию

Агент обучения с подкреплением, который учится проходить Geometry Dash (Steam) через захват экрана и инъекцию клавиатурного ввода. Без RL-фреймворков — всё написано с нуля на PyTorch.

## Требования

- Windows 10/11 (pydirectinput требует Windows)
- Python 3.12 или новее
- Geometry Dash (Steam, версия 2.2)
- GPU не требуется — работает на CPU

## Установка

```bash
git clone <repo-url>
cd gd-rl-agent
pip install -e .
```

## Шаг 1 — Калибровка

Запусти Geometry Dash в **оконном режиме** и начни Stereo Madness в **режиме практики (practice mode)**.

```bash
python cli.py calibrate
```

Команда выполнит следующее:
1. Автоматически найдёт окно GD
2. Измерит FPS захвата и задержку ввода
3. Сделает скриншот — открой `calibration_frame.png` и введи координаты пикселей по запросу
4. Сохранит конфиг в `configs/agent_config.json`

**Значения для ввода:**
- `player_right_x` — X-пиксель правого края спрайта куба
- `ground_y` — Y-пиксель верхней границы плитки земли
- `cube_height_px` — высота куба в пикселях (измерь на скриншоте)

## Шаг 2 — Проверка vision pipeline

```bash
python cli.py debug --duration 10
```

Следи за выводимыми векторами признаков и проверяй `debug_frame.png`. Зелёный прямоугольник — зона сканирования ROI, красная вертикаль — обнаруженное препятствие, оранжевая — яма. Если детекция не работает, нужно эмпирически подобрать HSV-пороги в `vision/obstacle_detector.py`.

## Шаг 3 — Обучение

Включи **Auto-Retry** в настройках GD. Запусти Stereo Madness в режиме практики и поставь чекпоинт в самом начале уровня (клавиша `Z`). Затем:

```bash
# Vanilla DQN
python cli.py train --episodes 500 --run-name vanilla

# Double DQN
python cli.py train --episodes 500 --double --run-name double

# Dueling DQN
python cli.py train --episodes 500 --dueling --run-name dueling
```

Логи сохраняются в `runs/<run-name>/train_log.csv`. Чекпоинты сохраняются каждые 50 эпизодов.

**Важно:** после первого запуска проверь реальный FPS в логах и пересчитай `max_episode_steps` в `configs/agent_config.json`, если он значительно отличается от значения при калибровке.

## Шаг 4 — Наблюдение и оценка

```bash
# Наблюдать за игрой агента (без обновления весов)
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 5

# Оценить среднюю награду и процент успеха
python cli.py eval --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 20
```

## Справочник конфигурации

Все настройки хранятся в `configs/agent_config.json` (создаётся командой `calibrate`).

| Параметр | Значение по умолчанию | Описание |
|---|---|---|
| `episode.max_episode_steps` | 250 | Шагов на эпизод — откалибровать с запущенным DQN |
| `episode.auto_retry` | true | true, если в GD включён Auto-Retry |
| `reward.r_step` | 0.01 | Награда за каждый выживший шаг |
| `reward.r_death` | -1.0 | Штраф за смерть |
| `reward.r_completion` | 5.0 | Бонус за достижение конца сегмента |
| `reward.penalize_useless_jumps` | true | Штрафовать прыжки без препятствия впереди |
| `reward.useless_jump_threshold` | 0.35 | Порог дистанции для "бесполезного" прыжка |
| `hp.use_double_dqn` | false | Включить Double DQN |
| `hp.use_dueling_dqn` | false | Включить Dueling DQN |
| `hp.use_obstacle_lookahead` | false | Добавить второе препятствие в вектор признаков |

## Структура проекта

```
gd-rl-agent/
├── capture/          # Захват экрана mss, определение окна
├── input/            # Инъекция прыжка через pydirectinput
├── vision/           # HSV-детекция препятствий/ям/смерти, State extractor
├── env/              # Цикл среды (reset/step/close)
├── agent/            # QNetwork, ReplayBuffer, Trainer
├── configs/          # Pydantic-модели конфигурации
├── cli.py            # Typer CLI (calibrate/debug/train/watch/eval)
└── tests/            # Unit-тесты agent/ (без реальной игры)
```

## Запуск тестов

```bash
python -m pytest tests/ -v
```

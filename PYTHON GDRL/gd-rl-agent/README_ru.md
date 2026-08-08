# GD RL Agent

Агент обучения с подкреплением, который учится проходить Geometry Dash (Steam) через захват экрана и инъекцию клавиатурного ввода. Учебный проект — без RL-фреймворков, всё написано с нуля на PyTorch.

## Ключевые ограничения

- Только CPU (Intel i7-1225U, GPU не требуется)
- Реальное время — ускорение симуляции и параллелизация невозможны
- Бинарное действие: прыжок / нет прыжка (только режим куба)
- Обучение на Stereo Madness 0–29% (сегмент куба, без порталов)

## Быстрый старт

```bash
pip install -e .
python cli.py calibrate        # найти окно, измерить задержку, сохранить конфиг
python cli.py debug            # визуально проверить vision pipeline
python cli.py train --episodes 500 --run-name vanilla
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt
```

## Варианты DQN

```bash
python cli.py train --no-double --no-dueling --run-name vanilla
python cli.py train --double    --no-dueling --run-name double
python cli.py train --no-double --dueling    --run-name dueling
```

## Архитектура

```
gd-rl-agent/
├── capture/       # захват экрана mss, определение окна
├── input/         # инъекция прыжка pydirectinput, измерение задержки
├── vision/        # HSV-детекция препятствий/ям/смерти, State extractor
├── env/           # цикл среды: reset() / step() / close()
├── agent/         # QNetwork, ReplayBuffer, Trainer
├── configs/       # Pydantic v2 модели конфигурации
├── cli.py         # Typer CLI: calibrate / debug / train / watch / eval
└── tests/         # unit-тесты agent/ (без реальной игры)
```

## Формула награды

```
r_t = +0.01   за каждый выживший шаг
    + (-1.0)  за смерть
    + (+5.0)  за завершение сегмента
    + (-0.02) за бесполезный прыжок (на земле, нет препятствия в радиусе порога)
γ = 0.99
```

## Вектор признаков (4 float)

| Признак | Диапазон | Сентинел |
|---|---|---|
| `dist_obstacle` | [0, 1] | 1.0 |
| `dist_pit` | [0, 1] | 1.0 |
| `obstacle_height` | [0, 1] | 0.0 |
| `is_grounded` | {0, 1} | — |

## Тесты

```bash
python -m pytest tests/ -v
```

Полное руководство по установке и калибровке — в `INSTRUCTION_ru.md`.

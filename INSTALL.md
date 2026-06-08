# Установка и запуск Tetris RL

Этот документ содержит практические инструкции по установке, тестированию,
обучению, оценке и визуализации проекта.

Описание идеи проекта и принципов работы агентов находится в
[README.md](README.md).

## Требования

- Python 3.14 или новее;
- macOS, Linux или Windows;
- `pip` или `uv`.

Для визуализации используется `pygame-ce`: он сохраняет привычный
`import pygame`, но корректно поддерживает Python 3.14.

Проверить версию Python:

```bash
python3 --version
```

Если установлена версия ниже 3.14, загрузите актуальный Python с
[python.org](https://www.python.org/downloads/) или используйте `uv`.

## Установка через venv и pip

Создайте виртуальное окружение из корня проекта:

```bash
python3.14 -m venv .venv
```

Активируйте его на macOS или Linux:

```bash
source .venv/bin/activate
```

На Windows:

```powershell
.venv\Scripts\activate
```

Установите зависимости и проект:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Установка через uv

Если установлен [uv](https://docs.astral.sh/uv/), зависимости можно
подготовить командой:

```bash
uv sync --python 3.14
```

Последующие команды можно запускать внутри окружения через `uv run`:

```bash
uv run python -m pytest
```

## Запуск тестов

Полный набор тестов:

```bash
python -m pytest
```

Подробный вывод:

```bash
python -m pytest -v
```

Запуск отдельного файла:

```bash
python -m pytest tests/test_env.py -v
python -m pytest tests/test_dqn_agent.py -v
python -m pytest tests/test_trainer.py -v
```

## Быстрая проверка training loop

Перед долгим обучением рекомендуется запустить smoke-конфигурацию:

```bash
python scripts/train.py \
  --config configs/smoke.yaml \
  --run-dir runs/smoke \
  --checkpoint-dir checkpoints/smoke
```

Она запускает два коротких эпизода и проверяет, что:

- training loop работает;
- loss вычисляется;
- метрики записываются;
- checkpoints сохраняются.

## Полное обучение DQN

Запустить обучение на `3000` эпизодах:

```bash
python scripts/train.py --config configs/default.yaml
```

По умолчанию создаются:

```text
runs/<timestamp>/metrics.csv
checkpoints/episode_100.pt
checkpoints/episode_200.pt
...
checkpoints/best.pt
checkpoints/final.pt
```

Указать отдельные директории:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --run-dir runs/experiment_01 \
  --checkpoint-dir checkpoints/experiment_01
```

Выбрать PyTorch-устройство:

```bash
python scripts/train.py --config configs/default.yaml --device cpu
python scripts/train.py --config configs/default.yaml --device mps
```

Доступные аргументы:

```bash
python scripts/train.py --help
```

## Конфигурация обучения

Основные настройки находятся в `configs/default.yaml`:

```yaml
episodes: 3000
gamma: 0.99
learning_rate: 0.001
batch_size: 128
replay_buffer_size: 50000
min_replay_size: 1000
target_update_every: 500
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.995
checkpoint_every: 100
eval_every: 100
eval_episodes: 10
max_steps_per_episode: 5000
seed: 42
```

Для нового эксперимента создайте копию YAML-файла и измените нужные значения.

## Оценка агентов

Сравнить Random Agent и Heuristic Agent:

```bash
python scripts/evaluate.py --episodes 30 --seed 42
```

Добавить обученный DQN:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --episodes 10 \
  --seed 42 \
  --max-steps 1000
```

Выбрать устройство для DQN:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --device mps
```

Все аргументы:

```bash
python scripts/evaluate.py --help
```

## Визуализация

Посмотреть Random Agent:

```bash
python scripts/play.py --agent random
```

Посмотреть Heuristic Agent:

```bash
python scripts/play.py --agent heuristic
```

Посмотреть обученный DQN:

```bash
python scripts/play.py \
  --agent dqn \
  --checkpoint checkpoints/best.pt
```

Настроить seed, скорость падения и размер клетки:

```bash
python scripts/play.py \
  --agent dqn \
  --checkpoint checkpoints/best.pt \
  --seed 42 \
  --speed 30 \
  --cell-size 30
```

Управление:

| Клавиша | Действие |
| --- | --- |
| `Space` | пауза или продолжение |
| `R` | перезапуск с тем же seed |
| `Up` / `Right` | увеличить скорость |
| `Down` / `Left` | уменьшить скорость |
| `Esc` | закрыть окно |

Все аргументы:

```bash
python scripts/play.py --help
```

## Построение графиков

Создать графики по сохранённому `metrics.csv`:

```bash
python scripts/plot_metrics.py --run runs/20260605_130715
```

В выбранной директории появятся:

```text
reward.png
lines.png
loss.png
```

Все аргументы:

```bash
python scripts/plot_metrics.py --help
```

## Запуск примеров baseline-агентов

В корне проекта находятся простые текстовые примеры:

```bash
python run_random.py
python run_heuristic.py
```

## Основные директории

```text
configs/       конфигурации обучения
checkpoints/   сохранённые модели
runs/          метрики и графики
scripts/       CLI-команды
tests/         автоматические тесты
tetris_rl/     код проекта
```

## Частые проблемы

### Команда `python` использует старую версию

Используйте `python3.14` при создании окружения или запускайте проект через
`uv`.

### Не найден модуль `tetris_rl`

Установите проект в editable-режиме:

```bash
python -m pip install -e .
```

### Не найден checkpoint

Сначала запустите обучение или укажите существующий файл:

```bash
python scripts/play.py --agent dqn --checkpoint checkpoints/best.pt
```

### Pygame не открывает окно

Проверьте, что команда запускается в окружении с графическим интерфейсом.
Удалённые серверы и headless-контейнеры обычно не могут открыть Pygame-окно.

### Matplotlib создаёт font cache

При первом запуске построения графиков Matplotlib может некоторое время
создавать кэш шрифтов. Это нормальное поведение.

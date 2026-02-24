# CV playground: эксперименты с CIFAR‑10 и DL‑архитектурами

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR10-orange?style=for-the-badge)

## Описание проекта

`CV playground` — это обучающий песочница‑проект, в котором на реальной задаче классификации CIFAR‑10 я последовательно исследую архитектуры, пайплайны и метрики глубокого обучения для компьютерного зрения.

В текущей версии ноутбука реализована серия экспериментов по **классификации CIFAR‑10 (10 классов, 32×32×3)**:
от простого **MLP‑бейслайна** до свёрточных архитектур (**SimpleCNN**, **SimpleResNet**, **MiniVGG**) и **transfer learning с предобученным ResNet18**.

Основные цели:

- собрать **чистый, переиспользуемый тренинг‑луп** (данные → модель → обучение → метрики → логи);
- отладить **аугментации** и **нормализацию** под CIFAR‑10;
- сравнить влияние архитектур: MLP ↔ CNN ↔ ResNet/VGG ↔ transfer learning (ResNet18).

Цепочка пайплайна: **подготовка данных CIFAR‑10 на диске → аугментации и нормализация → выбранная архитектура (MLP / SimpleCNN / SimpleResNet / MiniVGG / ResNet18‑head) → обучение/валидация → сохранение чекпоинтов и кривых обучения в `runs/`**.

## Навыки и инструменты

**Сферы деятельности:**

Computer Vision, Image Classification, Deep Learning, CIFAR‑10, эксперименты с архитектурами (MLP/CNN/ResNet/VGG), Transfer Learning, reproducible research (Config, seed).

**Инструменты:**

PyTorch, torchvision (включая `torchvision.models.resnet18` и `ResNet18_Weights`), NumPy, dataclasses, tqdm, Matplotlib, Python OOP/utility‑функции.

**Ключевые этапы разработки:**

*   **Конфигурация и воспроизводимость:** `Config` через `@dataclass` (exp_name, model, use_bn, seed, epochs, batch_size, lr, weight_decay, val_split, num_workers). Функции `set_seed` и `make_run_dir` фиксируют случайность и сохраняют `config.json` в каталоге экспа.
*   **Подготовка данных CIFAR‑10:** чтение данных с диска через `datasets.ImageFolder` из `./data/train` и `./data/test`. Разделение на train/val (`train/val sizes: 45000 / 5000`) с фиксированным `Generator` для воспроизводимых сплитов.
*   **Аугментации и нормализация:** пайплайн `transforms.Compose` с `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, конвертацией в тензоры и нормализацией по `CIFAR10_MEAN` и `CIFAR10_STD`. Отдельные трансформации для train/val/test.
*   **Dataloaders:** аккуратная сборка `DataLoader` для train/val/test с учетом `pin_memory` на GPU, `num_workers`, `shuffle` и батч‑размеров из `CFG`.
*   **Метрики и лупы обучения:** реализация `accuracy_top1`, `train_one_epoch`, `validate` и общей функции `fit` с `tqdm`, логированием лоссов/аккуратности и, при необходимости, `ReduceLROnPlateau` по валидационному лоссу.
*   **Архитектуры:**
    * `MLPBaseline`: flatten → два полносвязных слоя с ReLU → логиты на 10 классов.
    * `SimpleCNN`: три `ConvBlock` (Conv2d + BatchNorm2d + ReLU) с `MaxPool2d` и линейным классификатором.
    * `SimpleResNet`: `ResidualBlock`‑и с shortcut‑веткой, три residual‑слоя, `AdaptiveAvgPool2d` и полносвязный слой.
    * `MiniVGG`: VGG‑подобные блоки (`VGGBlock`) с несколькими Conv+BN+ReLU и MaxPool, затем `AdaptiveAvgPool2d` и линейный слой.
*   **Transfer learning с ResNet18:**
    * Использование `resnet18(weights=ResNet18_Weights.DEFAULT)` как фиче‑экстрактора.
    * Заморозка большей части бэкбона, обучение только `layer4` и небольшой MLP‑головы с Dropout.
    * Разные learning rate для головы и `layer4`, `ReduceLROnPlateau` по val‑лоссу.
    * Отдельные `train_loader_pre` / `val_loader_pre` для обучения на предвычисленных фичах.
*   **Логи и визуализация:** функция `plot_history` строит и сохраняет кривые `loss` и `accuracy` (train/val) в папку запуска в `runs/`. Лучший чекпоинт модели сохраняется как `best.pt` по метрике `val_acc`.

## Выводы и результаты

По результатам запусков на 10 эпох на CIFAR‑10:

- собран **полный рабочий пайплайн**: от чтения данных и аугментаций до обучения разных архитектур и сохранения экспериментов;
- базовый `MLPBaseline` даёт около **~0.45 val accuracy** и служит реалистичным MLP‑бейслайном;
- простая свёрточная модель **`SimpleCNN`** поднимает качество до **~0.76 val accuracy**;
- residual‑архитектура **`SimpleResNet`** выходит примерно на **~0.81 val accuracy**;
- VGG‑подобная **`MiniVGG`** достигает порядка **~0.84 val accuracy**;
- при **transfer learning с предобученным `ResNet18`** (замороженный бэкбон, обучение `layer4` + MLP‑головы с Dropout) удаётся получить **~0.89–0.90 val accuracy**: модель быстро переобучается на train, но вал‑качество держится стабильно.

Код тренинга и визуализации вынесен в отдельные функции, что упрощает последующие эксперименты (можно менять модель/лосс/метрики при сохранении общего контура и логики `runs/`).

Дальнейшие шаги: полный fine‑tuning ResNet/PreAct ResNet, эксперименты с регуляризацией (Dropout, weight decay, label smoothing), scheduler’ами и, в перспективе, сравнение с более тяжёлыми CNN и ViT‑подобными моделями.



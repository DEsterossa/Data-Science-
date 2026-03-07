# PH2: бинарная сегментация кожных образований

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch)
![PH2](https://img.shields.io/badge/Dataset-PH2-orange?style=for-the-badge)

## Описание проекта

`PH2` — это небольшой **segmentation playground** для задачи **бинарной сегментации кожных образований (lesion vs background)** на дерматоскопических изображениях из датасета **PH2**.

В ноутбуке последовательно собирается и исследуется базовый пайплайн сегментации:

- подготовка и sanity‑проверка датасета (dataset, dataloaders, визуализация изображений и масок);
- реализация базовой архитектуры **U‑Net, обучаемой с нуля**;
- обучение и анализ метрик сегментации (**Dice, IoU**, как soft, так и hard);
- сравнение **функций потерь** (BCE, Dice, BCE+Dice);
- эксперименты с **шириной модели (model capacity)**;
- исследование влияния **порога бинаризации (threshold)**;
- сравнение baseline‑модели с **U‑Net с предобученным encoder‑ом (ResNet34)**;
- визуальный анализ ошибок и сложных кейсов.

Цель экспериментов — понять, какие компоненты пайплайна (архитектура, loss, threshold, capacity, pretrained encoder) дают наибольший вклад в качество сегментации на небольшом медицинском датасете.

## Навыки и инструменты

**Сферы деятельности:**

Computer Vision, Medical Imaging, Semantic Segmentation, Binary Segmentation, PH2, U‑Net, Transfer Learning, экспериментальный анализ (loss/threshold/capacity), reproducible research.

**Инструменты:**

PyTorch, `segmentation_models_pytorch` (U‑Net с ResNet‑encoder‑ом), torchvision, PIL, NumPy, Matplotlib, Python OOP/utility‑функции.

**Ключевые этапы разработки:**

*   **Конфигурация и воспроизводимость:** словарь `CFG` (seed, img_size, batch_size, num_workers, val_ratio, pin_memory). Функция `seed_everything` фиксирует все генераторы случайных чисел и настройки cuDNN.
*   **Подготовка данных PH2:** класс `PH2SegDataset` собирает пары `(изображение, маска)` по структуре `PH2Dataset/PH2 Dataset images/IMD*/..._Dermoscopic_Image / ..._lesion`. Внутри реализованы:
    * корректный `resize_with_pad` без искажения пропорций (маски — через NEAREST, изображения — через BILINEAR);
    * перевод маски в бинарный формат {0,1} и тензор `1×H×W`.
*   **Dataloaders:** разбиение на train/val по `val_ratio` с фиксированным сидом, сборка `DataLoader` с `batch_size`, `num_workers` и `pin_memory` из `CFG`.
*   **Модель‑baseline (U‑Net с нуля):** компактная `UNetSmall` с настраиваемой шириной (`base_ch`), обучается с нуля на PH2 без внешнего pretrain.
*   **Pretrained модель:** U‑Net на основе `segmentation_models_pytorch` с encoder‑ом ResNet34, инициализированным предобученными весами; decoder и segmentation‑голова обучаются на PH2.
*   **Функции потерь и метрики:**
    * отдельные реализации BCE‑loss, Dice‑loss;
    * комбинированный `bce_dice_loss_fn(logits, target, alpha=1.0, beta=1.0)`;
    * вычисление soft/hard Dice и IoU на валидации.
*   **Тренинг‑луп:** функция `fit` с:
    * обучением модели и валидацией по эпохам;
    * логированием `train_loss`, `val_loss`, `soft_dice`, `soft_iou`, `hard_dice`, `hard_iou`;
    * `ReduceLROnPlateau` по `val_loss` для стабилизации обучения.
*   **Эксперименты:**
    * сравнение loss‑функций (BCE, Dice, BCE+Dice);
    * эксперименты с шириной U‑Net (`base_ch = 16, 32, 64`);
    * подбор порога бинаризации (threshold ~0.7);
    * сравнение baseline‑U‑Net и U‑Net с pretrained encoder‑ом.

## Выводы и результаты

### Baseline U‑Net (обучение с нуля)

- Небольшой U‑Net (`UNetSmall`, обученный с нуля на PH2) показывает:
  - **hard Dice ≈ 0.88**
  - **hard IoU ≈ 0.79**
- Модель адекватно сегментирует большинство случаев, но:
  - часто ошибается на сложных/атипичных примерах;
  - иногда предсказывает чрезмерно большие области или «ломает» форму lesion;
  - чувствительна к качеству признаков encoder‑а, который обучается с нуля на маленьком датасете.

### U‑Net с предобученным encoder‑ом (ResNet34)

- Подмена encoder‑а на предобученный ResNet34 даёт заметный прирост:
  - **hard Dice ≈ 0.95**
  - **hard IoU ≈ 0.90**
- Pretrained encoder даёт более устойчивые визуальные признаки (контуры, текстуры, границы), что:
  - улучшает локализацию lesion;
  - помогает лучше восстанавливать форму и границы;
  - снижает количество грубых ошибок на сложных примерах.

### Влияние функции потерь

Сравнение разных loss‑функций для baseline‑U‑Net:

| Loss        | Hard Dice |
|------------|-----------|
| BCE        | ~0.69     |
| Dice       | ~0.80     |
| BCE + Dice | **~0.84** |

Комбинация **BCE + Dice** даёт наилучший баланс между пиксельной точностью и глобальным перекрытием маски.

### Влияние ширины модели (capacity)

Эксперимент с `base_ch = 16, 32, 64` показал, что увеличение ширины почти не улучшает качество:

- `base_ch=16 → best hard Dice ≈ 0.86`
- `base_ch=32 → best hard Dice ≈ 0.84`
- `base_ch=64 → best hard Dice ≈ 0.85 (при этом модель склонна к переобучению)`

Ограничение baseline‑модели связано не с capacity, а с качеством признаков encoder‑а, обучаемого с нуля.

### Влияние threshold

Небольшой прирост по hard‑метрикам достигается при увеличении порога бинаризации до **~0.7**. Эффект есть, но он значительно слабее, чем влияние выбора архитектуры и encoder‑а.

### Итог

Ключевой фактор, определяющий качество сегментации на PH2, — использование **предобученного encoder‑а**.
Даже небольшой U‑Net с pretrained backbone заметно превосходит модель, обученную с нуля на ограниченном количестве данных.

## Как запустить

- Откройте ноутбук `notebook.ipynb` в папке `ph2`.
- Убедитесь, что датасет PH2 распакован в `PH2Dataset/PH2 Dataset images/...` (структура, ожидаемая в `PH2SegDataset` — подпапка `IMD*` с `*_Dermoscopic_Image` и `*_lesion`).
- При необходимости скорректируйте пути к данным и размер изображений в `CFG` (особенно `img_size` и `batch_size`).
- Последовательно выполните ячейки:
  - блок `Setup` (импорты, `CFG`, `seed_everything`, `device`);
  - определение `resize_with_pad`, `PH2SegDataset` и сборка `DataLoader`‑ов;
  - определение моделей (baseline U‑Net и U‑Net с pretrained encoder‑ом);
  - эксперименты с loss‑функциями, шириной модели, threshold и pretrained encoder‑ом.
- Используйте сохранённые логи и визуализации (кривые loss/Dice/IoU и примеры предсказаний) для анализа качества сегментации и выбора финальной модели.


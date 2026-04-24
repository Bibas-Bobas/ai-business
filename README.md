# VIZARD / Ice Class Inpainting

Проект восстанавливает **пропуски (дырки) в картах классов льда** по снимкам Sentinel-1 (EW, GRDM) и производным композитам / результатам нейросетевой классификации. Задача: по входу «картинка с вырезом + маски + геоконтекст» предсказать недостающие пиксели так, чтобы на инференсе подставлять предсказание **только в области пропуска**, а валидные пиксели оставлять без изменений.

Дополнительные материалы в репозитории: `README_Statement.pdf` (постановка), `corrected_output_polygons.geojson` (районы СМП).

---

## Содержание

- [Что сделано в проекте](#что-сделано-в-проекте)
- [Структура репозитория](#структура-репозитория)
- [Структура датасета](#структура-датасета)
- [Зависимости](#зависимости)
- [Пайплайн: от сырых данных до модели](#пайплайн-от-сырых-данных-до-модели)
- [Обучение](#обучение)
- [Инференс и визуализация](#инференс-и-визуализация)
- [Дрейф: синтетика vs история снимков](#дрейф-синтетика-vs-история-снимков)
- [Деплой и продакшен](#деплой-и-продакшен)
- [Известные ограничения](#известные-ограничения)

---

## Что сделано в проекте

1. **`prepare.py`** — генерация обучающего датасета: нарезка патчей, маски дыр (геометрии `horizontal_tri` / `vertical_tri` / `irregular_shape`), маска суши, кодирование классов льда в один канал `uint8`, опционально канал **`drift_prev`** (синтетический «предыдущий кадр» через поле смещения + warp).
2. **`train_unet2.py`** — обучение **U-Net с энкодером ResNet34** (ImageNet), 7 входных каналов, семантическая сегментация с весом ошибки в дырке, игнор суши, AMP, **DDP** через `torchrun`.
3. **`train.py`** — альтернативный пайплайн (кастомный IceUNet, DDP, AMP, без RAM-кэша датасета); можно использовать как второй вариант экспериментов.
4. **`inference.py`** — оценка чекпойнта `train_unet2`, метрики по split, визуализация GT / маска / предсказание / восстановление.
5. **`mask_inpaint_predictor.py`** — класс **`IceMaskInpaintPredictor`**: загрузка `best.pt`, сбор 7 каналов из папок датасета, сохранение результата в GeoTIFF + опционально PNG-сравнение.
6. **`generate_drift_from_history.py`** — построение **`drift_prev`** и **`drift_uv`** по **нескольким реальным предыдущим снимкам** (время из имени Sentinel-1, выравнивание растров по геопривязке, временное усреднение, оценка смещения в пикселях).

---

![metrics](https://github.com/Bibas-Bobas/ai-business/blob/175e507349744208d537d9f3178600bfb99f9ba4/metrics.png)
![inference](https://github.com/Bibas-Bobas/ai-business/blob/4fcd32f950cc63cde76318dcd5fc3acf1b488524/inference.png)


## Структура репозитория

```
ai-business/
├── README.md                    # этот файл
├── README_Statement.pdf         # постановка / требования
├── prepare.py                   # подготовка датасета из исходных GeoTIFF
├── train_unet2.py               # основной скрипт обучения (ResNet34-UNet)
├── train.py                     # альтернативное обучение (IceUNet + DDP)
├── inference.py                 # метрики и визуализация для train_unet2
├── mask_inpaint_predictor.py    # инференс + сохранение GeoTIFF / PNG
├── generate_drift_from_history.py  # дрейф из стека снимков по времени и геометрии
├── corrected_output_polygons.geojson
├── best.pt / epoch_*.pt         # чекпойнты (не коммитить в git при большом размере)
├── dataset/                     # часто: внешний архив dataset.zip
│   └── dataset/                 # фактический корень с samples.csv (см. ниже)
├── Dataset_2025_IceClass/       # пример исходных сцен (если есть локально)
└── runs/                        # артефакты запусков (логи, картинки)
```

Если после распаковки `samples.csv` лежит по пути `dataset/dataset/samples.csv`, в командах указывайте **`--dataset-root dataset/dataset`** или **`--data-root dataset/dataset`** (для `train_unet2`).

---

## Структура датасета

Корень датасета (пример: `dataset/dataset/`) содержит:

| Файл / папка | Назначение |
|--------------|------------|
| `samples.csv` | Разметка: `split`, `version`, `sample_id`, `source_file`, `patch_index`, `time_norm`, `top`, `left`, `patch_size`, флаги `drift_prev`, `drift_uv` |
| `meta.csv` | Параметры генерации: `patch_size`, `target_gb`, `versions`, и т.д. |
| `train/`, `val/`, `test/` | Для каждого split — подпапки по варианту маски дырки |
| `<split>/<version>/original/` | Эталонная карта классов (одноканальный GeoTIFF, коды 0–255) |
| `<split>/<version>/masked/` | Вход с «дыркой» (в дырке нули) |
| `<split>/<version>/masks/` | Бинарная маска пропуска: 1 = восстановить |
| `<split>/<version>/land_mask/` | Суша: 1 = игнор в лоссе / контекст для сети |
| `<split>/<version>/drift_prev/` | Опционально: канал «предыдущего состояния» (синтетика из `prepare` или внешний `generate_drift_from_history`) |

**Размер патча по умолчанию:** `384×384` (см. `meta.csv`: `patch_size,384`).

**Классы льда:** задаются в `prepare.py` (`ICE_PALETTE`, по умолчанию `--num-classes 9`); значения в растре кодируются через `np.linspace(1, 255, num_classes)`.

Один логический **sample** = одинаковый stem имени файла во всех подпапках (`sample_id` в CSV).

---

## Зависимости

### Обязательные (минимум)

| Пакет | Назначение |
|--------|------------|
| Python | рекомендуется **3.10+** |
| `torch`, `torchvision` | обучение и инференс (`train_unet2` использует ResNet34) |
| `numpy` | массивы |
| `rasterio` | чтение/запись GeoTIFF |
| `tqdm` | прогресс |
| `Pillow` (`PIL`) | используется в `prepare.py` |
| `matplotlib` | визуализация в `inference.py`, `mask_inpaint_predictor.py` |

### Опционально

| Пакет | Назначение |
|--------|------------|
| `opencv-python` | плотный optical flow в `generate_drift_from_history.py` |
| `scikit-image` | запасная оценка сдвига (phase correlation), если OpenCV недоступен |

### Совместимость NumPy и PyTorch

Сборки PyTorch часто собраны под **NumPy 1.x**. При **NumPy 2.x** возможны сбои при инициализации CUDA. Рекомендация: `numpy<2` в окружении обучения/инференса, либо PyTorch-сборка под NumPy 2.

Установка из файла (см. ниже `requirements.txt`):

```bash
python -m pip install -r requirements.txt
```

Установка PyTorch с GPU: см. [pytorch.org/get-started](https://pytorch.org/get-started/locally/) (CUDA-версия должна совпадать с драйвером).

---

## Пайплайн: от сырых данных до модели

### 1. Подготовка датасета (`prepare.py`)

Пример:

```bash
python prepare.py ^
  --input-root path/to/Dataset_2025_IceClass ^
  --output-root path/to/dataset ^
  --land-vector path/to/russia.shp ^
  --patch-size 384 ^
  --target-gb 6 ^
  --num-classes 9 ^
  --save-drift-prev
```

Ключевые аргументы:

- `--input-root` — исходные сцены (GeoTIFF).
- `--output-root` — куда писать `train/val/test`, `samples.csv`, `meta.csv`.
- `--land-vector` / `--land-raster` — исключение суши (по проекту может быть `Russia` shapefile и т.п.).
- `--patch-size` — размер патча (по умолчанию 384).
- `--target-gb` — ориентир объёма датасета.
- `--save-drift-prev` — генерировать синтетический `drift_prev/`.
- `--versions` — какие формы дырок создавать (`horizontal_tri`, `vertical_tri`, `irregular_shape`).

После генерации проверьте `meta.csv` и наличие `samples.csv` в корне вывода.

### 2. Обучение

См. раздел [Обучение](#обучение).

### 3. Оценка и продакшен

См. [Инференс и визуализация](#инференс-и-визуализация) и [Деплой](#деплой-и-продакшен).

---

## Обучение

### Вариант A: `train_unet2.py` (рекомендуемый для текущего датасета)

Модель: **ResNet34UNet**, вход **7 каналов**:  
`masked`, `hole`, `land`, `drift`, `x`, `y`, `time_norm`.

**Один GPU:**

```bash
python train_unet2.py ^
  --data-root dataset/dataset ^
  --save-dir runs/ice_unet2 ^
  --num-classes 9 ^
  --epochs 60 ^
  --batch-size 8 ^
  --amp ^
  --precision bf16
```

**Несколько GPU (DDP):**

```bash
torchrun --nproc_per_node=4 train_unet2.py ^
  --data-root dataset/dataset ^
  --save-dir runs/ice_unet2 ^
  --num-classes 9 ^
  --epochs 60 ^
  --batch-size 8 ^
  --amp ^
  --precision bf16
```

Чекпойнты: `best.pt` (лучший mIoU на валидации), `last.pt`, `epoch_XXX.pt`. Внутри: `model`, `optimizer`, `args`, `epoch`, `best_metric`.

Важно: `--num-classes` должен совпадать с подготовкой данных (`prepare.py`: по умолчанию 9 ледовых классов в палитре; выход сети = `num_classes + 1` с фоном 0).

### Вариант B: `train.py` (альтернатива, кастомный UNet)

Многопроцессный запуск:

```bash
torchrun --standalone --nproc_per_node=4 train.py ^
  --dataset-root dataset/dataset ^
  --epochs 40 ^
  --batch-size 32 ^
  --class-values 1,33,64,96,128,160,192,223
```

Параметры классов и пути смотрите в `parse_args` внутри `train.py` (автоподбор воркеров, AMP, DDP).

---

## Инференс и визуализация

### `inference.py` (чекпойнт от `train_unet2`)

Корень датасета должен содержать `samples.csv` (часто это **`dataset/dataset`**).

```bash
python inference.py ^
  --checkpoint best.pt ^
  --dataset-root dataset/dataset ^
  --split val ^
  --metrics ^
  --num-samples 16 ^
  --output-dir runs/inference
```

- Без GPU: добавьте `--device cpu --no-amp`.
- Только метрики: `--metrics --no-vis`.

### `mask_inpaint_predictor.py`

Сохранение результата в GeoTIFF в формате, согласованном с исходными растрами, плюс опционально PNG:

```bash
python mask_inpaint_predictor.py ^
  --checkpoint best.pt ^
  --dataset-root dataset/dataset ^
  --split train ^
  --version vertical_tri ^
  --sample-id YOUR_SAMPLE_ID ^
  --output-tif runs/predictions/out.tif ^
  --output-vis runs/predictions/out.png
```

---

## Дрейф: синтетика vs история снимков

| Источник | Где задаётся | Смысл |
|----------|----------------|-------|
| Синтетика | `prepare.py --save-drift-prev` | Warp исходного патча случайным гладким полем — имитация движения льда между проходами |
| История | `generate_drift_from_history.py` | Несколько **реальных** предыдущих GeoTIFF по времени (даты в имени Sentinel-1), выравнивание на сетку текущего снимка, взвешивание по времени, опционально `*_drift_uv.tif` (u,v в пикселях) |

На **инференсе в бизнесе** канал `drift_prev` не появляется сам: его нужно либо сгенерировать (как в `prepare`), либо получить из истории (`generate_drift_from_history`), либо подать **нули**, если модель к этому устойчива.

Пример:

```bash
python generate_drift_from_history.py ^
  --scenes-dir Dataset_2025_IceClass ^
  --output-dir runs/drift_from_history ^
  --num-prev 4 ^
  --min-overlap 0.02 ^
  --tau-hours 48 ^
  --num-classes 9
```

---

## Деплой и продакшен

### Минимальный продакшен-пайплайн

1. **Вход:** GeoTIFF с картой классов с пропусками, бинарная маска дырки, маска суши, метаданные (CRS, transform), при необходимости — канал `drift_prev` и/или нормализованное время.
2. **Нормализация:** как в `train_unet2.VizardDataset`: `masked/255`, маски `0/1`, `drift/255`, координаты `[-1,1]`, канал времени константой по изображению.
3. **Модель:** загрузка весов из `best.pt` (ключ `model`), класс `ResNet34UNet` с `in_channels=7`, `num_classes=num_classes+1`.
4. **Постобработка:** как в обучении/инференсе — подстановка предсказания **только в маске пропуска**; вне маски оставить исходные значения.
5. **Выход:** GeoTIFF с тем же профилем, что у входного растра (см. `mask_inpaint_predictor.py` / `rasterio`).

### Окружение

- Зафиксировать версии: `python`, `torch`, `cuda`, `numpy`, `rasterio` (файл `requirements.txt`).
- Для сервера с GPU: образ Docker с CUDA + установленным PyTorch или conda-окружение.

### Масштабирование инференса

- Большие сцены: нарезка на тайлы с overlap и смешивание на границах (логика не включена в репозиторий по умолчанию — типичный следующий шаг).
- Батчинг на GPU; для CPU — `--device cpu`, отключить AMP.

### Секреты и данные

- Не публиковать закрытые GeoTIFF и веса в открытый репозиторий без политики организации.


---

## Быстрый старт (чек-лист)

1. `python -m pip install -r requirements.txt` (+ PyTorch с CUDA при необходимости).
2. Подготовить/распаковать датасет так, чтобы существовал `dataset/dataset/samples.csv`.
3. Обучить: `train_unet2.py` + `torchrun` при нескольких GPU.
4. Проверить: `inference.py --metrics` на `val`.
5. Экспорт предсказания в TIFF: `mask_inpaint_predictor.py`.


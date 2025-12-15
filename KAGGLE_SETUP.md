# Kaggle Setup Guide

Этот документ описывает, как запустить Score-Entropy-Discrete-Diffusion проект на Kaggle.

## Изменения для совместимости с Kaggle

1. **Удален flash-attn из requirements.txt** - Kaggle не поддерживает flash-attn на доступных GPU
2. **Создана конфигурация для Kaggle** (`configs/config_kaggle.yaml`)
   - Меньший batch size (64 вместо 512)
   - Gradient accumulation для эффективного увеличения размера батча
   - Меньше итераций для быстрого тестирования

3. **Обновлен kaggle_train.py** с правильной конфигурацией

## Workflow с Kaggle

### Локально (на вашей машине)

1. Отредактируйте код локально
2. Закоммитьте и пушьте в GitHub:
```bash
git add .
git commit -m "Fix: remove flash-attn, add kaggle config"
git push origin main
```

### В Kaggle Notebook

1. Создайте новый notebook в Kaggle
2. Добавьте ячейку для клонирования вашего репозитория:

```python
import os
os.system('git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git')
os.chdir('YOUR_REPO')
```

3. Установите зависимости:

```python
os.system('pip install -q -r requirements.txt')
```

4. Запустите обучение:

```python
os.system('python kaggle_train.py')
```

## Инструкции по запуску

### Локально (для тестирования)

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить тренировку с Kaggle конфигом
python kaggle_train.py

# Или с оригинальной конфигом
python train.py
```

### На Kaggle

В Kaggle Notebook используйте:

```python
# Клонирование и установка
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -q -r requirements.txt

# Запуск тренировки
!python kaggle_train.py
```

## Параметры конфигурации

### config.yaml (оригинальная конфигурация)
- **batch_size**: 512
- **n_iters**: 1,300,001
- **snapshot_freq**: 50,000
- Для мощных многопроцессорных систем

### config_kaggle.yaml (для Kaggle)
- **batch_size**: 64
- **accum**: 8 (градиент аккумуляция)
- **n_iters**: 100,000 (для тестирования)
- **snapshot_freq**: 5,000
- Оптимизирован для ограниченной памяти GPU

## Что было изменено в коде

### model/transformer.py
- Добавлена поддержка fallback на стандартный `F.scaled_dot_product_attention` когда flash-attn недоступна
- `_HAS_FLASH_ATTN` флаг контролирует использование flash-attn

### model/rotary.py
- Добавлена обработка import error для flash_attn rotary
- Использует стандартную rotary embeddings как fallback

### requirements.txt
- Удален `flash-attn==2.2.2`

## Отладка

Если возникают проблемы:

1. **CUDA out of memory**: Уменьшите `batch_size` в конфиге
2. **ImportError для flash-attn**: Это ожидается, код использует fallback
3. **Данные не загружаются**: Убедитесь, что Kaggle может скачать OpenWebText
4. **Проблемы с NCCL**: На Kaggle используется single GPU, NCCL не нужен

## Next Steps

После успешного запуска:

1. Проверьте логи в `exp_local/openwebtext/*/logs`
2. Проверьте checkpoints в `exp_local/openwebtext/*/checkpoints`
3. Отрегулируйте параметры в конфиге по мере необходимости
4. Пушьте изменения в GitHub и синхронизируйте в Kaggle

## Полезные ссылки

- [Статья (article.pdf)](article.pdf)
- [README.md](README.md)

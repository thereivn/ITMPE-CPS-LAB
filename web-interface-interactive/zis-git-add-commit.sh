#!/bin/bash

BUILD_FILE="build_number.txt"
if [[ ! -f "$BUILD_FILE" ]]; then
  echo "1" > "$BUILD_FILE"
fi

BUILD_NUMBER=$(cat "$BUILD_FILE")
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# Получаем список изменённых и добавленных файлов (unstaged)
CHANGED_FILES=$(git diff --name-only --diff-filter=AM)

if [[ -z "$CHANGED_FILES" ]]; then
  # Если нет unstaged изменений, проверим staged изменения
  CHANGED_FILES=$(git diff --cached --name-only --diff-filter=AM)
fi

if [[ -z "$CHANGED_FILES" ]]; then
  echo "Нет изменений для коммита."
  exit 0
fi

# Добавляем все изменённые и добавленные файлы в индекс (на всякий случай)
git add $CHANGED_FILES

# Формируем сообщение коммита с номером сборки, временем и списком изменённых файлов
COMMIT_MESSAGE="Build #$BUILD_NUMBER at $CURRENT_TIME

Изменено:
$CHANGED_FILES
"

git commit -m "$COMMIT_MESSAGE"

# Увеличиваем номер сборки на 1 и сохраняем обратно
NEXT_BUILD_NUMBER=$((BUILD_NUMBER + 1))
echo "$NEXT_BUILD_NUMBER" > "$BUILD_FILE"

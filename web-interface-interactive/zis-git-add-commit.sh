#!/bin/bash

BUILD_FILE="build_number.txt"
if [[ ! -f "$BUILD_FILE" ]]; then
  echo "1" > "$BUILD_FILE"
fi

BUILD_NUMBER=$(cat "$BUILD_FILE")
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
COMMIT_MESSAGE="Build #$BUILD_NUMBER at $CURRENT_TIME"

git add .
git commit -m "$COMMIT_MESSAGE"

# Увеличиваем номер сборки на 1 и сохраняем обратно
NEXT_BUILD_NUMBER=$((BUILD_NUMBER + 1))
echo "$NEXT_BUILD_NUMBER" > "$BUILD_FILE"

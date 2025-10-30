#!/bin/bash

IMAGE_NAME="thereivn/python-base-for-streamlit-app:0.0.1"

if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "Образ $IMAGE_NAME не найден локально. Запускаю сборку..."
  docker build -f Z.PythonBase.Dockerfile -t $IMAGE_NAME .
else
  echo "Образ $IMAGE_NAME уже существует. Пропускаю сборку."
fi

docker build -f Z.Streamlit.Dockerfile -t thereivn/reliability-streamlit-app:0.0.5 .
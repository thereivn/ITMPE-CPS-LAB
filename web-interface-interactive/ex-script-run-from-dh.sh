#!/bin/bash

docker kill reliability-streamlit-app && docker rm reliability-streamlit-app

docker run -d --name reliability-streamlit-app -p 8501:8501 thereivn/reliability-streamlit-app:0.0.4

echo "Интерфейс будет доступен в браузере по адресу:"
echo ""
echo "http://localhost:8501"
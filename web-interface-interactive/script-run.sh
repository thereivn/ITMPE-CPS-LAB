#!/bin/bash

docker run -d -p 8501:8501 reliability_streamlit_app

echo "Интерфейс будет доступен в браузере по адресу:"
echo ""
echo "http://localhost:8501"
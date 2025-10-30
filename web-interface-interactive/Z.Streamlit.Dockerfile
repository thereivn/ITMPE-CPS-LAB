FROM thereivn/python-base-for-streamlit-app:0.0.1
COPY main.py /app/
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
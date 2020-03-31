FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "-u", "app.py"]

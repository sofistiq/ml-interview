# 1. Step: Install dependencies
FROM tensorflow/tensorflow:latest-gpu-py3 as Dependencies
WORKDIR /app
RUN pip install tensorflow-cpu

# 2. Stage: Run
FROM Dependencies as Run
WORKDIR /app
COPY src .
CMD ["python", "main.py"]

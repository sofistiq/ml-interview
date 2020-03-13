# 1. Step: Install dependencies
FROM tensorflow/tensorflow:latest-gpu-py3 as Dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Stage: Run
FROM Dependencies as Run
WORKDIR /app
COPY . .
EXPOSE 5000
CMD ["python", "src/main.py"]

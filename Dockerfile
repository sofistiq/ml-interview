# 1. Stage:
FROM python:3-alpine
WORKDIR /app
COPY src .
CMD ["python", "main.py"]

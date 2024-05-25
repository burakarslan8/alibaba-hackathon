FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --fix-missing --no-install-recommends \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

CMD ["python", "app.py"]
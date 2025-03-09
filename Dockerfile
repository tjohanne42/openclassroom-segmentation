FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y git-lfs && git lfs install

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py best_model.keras ./

ENV GUNICORN_CMD_ARGS="--workers=2 --timeout 120 --access-logfile -"

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
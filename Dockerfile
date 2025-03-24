FROM python:3.12.7-slim

RUN apt-get update  \
    && apt-get install -y --no-install-recommends portaudio19-dev python3-dev gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /workspace
ENV PYTHONPATH="${PYTHONPATH}:."

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

CMD ["python", "client.py"]
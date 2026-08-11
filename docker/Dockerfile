FROM python:3.11-slim

# Tesseract for the zero-dependency OCR baseline; skip paddleocr/easyocr
# system deps here to keep the image small — install the `ocr` extra
# manually inside the container if you need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -e ".[ocr]"

ENTRYPOINT []
CMD ["slmbench", "list-models"]

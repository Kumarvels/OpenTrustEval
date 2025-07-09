FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt fastapi uvicorn[standard] torch transformers pillow psutil

EXPOSE 8000

CMD ["uvicorn", "ote_api:app", "--host", "0.0.0.0", "--port", "8000"]
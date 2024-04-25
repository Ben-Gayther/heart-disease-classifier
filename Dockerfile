FROM python:3.12.2

ENV SERVER_HOST 0.0.0.0
ENV HTTP_PORT 8000

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${HTTP_PORT}

CMD uvicorn app:router --host ${SERVER_HOST} --port ${HTTP_PORT} --app-dir heart-disease-classifier
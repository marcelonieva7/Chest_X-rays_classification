FROM python:3.11-slim
RUN pip install pipenv

WORKDIR /app
COPY ["container/Pipfile", "container/Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "x-rays-model.tflite", "./"]

COPY templates ./templates

COPY static ./static

COPY server ./server

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
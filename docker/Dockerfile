FROM python:3.9

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app ./src

COPY ./docker/entrypoint.sh ./

RUN mkdir -p ./src/models

RUN chmod +x entrypoint.sh

ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
FROM python:3.12-slim

RUN apt update && apt install -y curl iputils-ping
RUN apt update && apt install -y gcc python3-dev python3-pip

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
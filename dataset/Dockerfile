FROM python:3.7

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
WORKDIR /workspace/

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

# COPY Code and config
COPY dataset dataset
COPY data data
RUN chown -R 1147:1003 .
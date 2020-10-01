FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
WORKDIR /workspace/

COPY model/model_requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt
RUN python -m spacy download en

# COPY Code and config
COPY website website
COPY util util
COPY data data
RUN chown -R 1147:1003 .
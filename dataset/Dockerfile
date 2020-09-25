FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

# docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/conference-matcher:latest .
# docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/conference-matcher:latest
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

# COPY Code and config
COPY dataset dataset
RUN chown -R 1147:1003 /workdir
FROM python:3.9-slim-buster

LABEL MAINTAINER="Christophe Karam <christophe.karam@grenoble-inp.org>"

ENV DEBIAN_FRONTEND=noninteractive

ARG USER=prognostics
RUN useradd -ms /bin/bash $USER

ENV SHELL=/bin/bash
ENV HOME=/home/$USER
ENV PATH=$HOME/.local/bin:$PATH


USER root
RUN apt-get update && apt-get -yq install curl

WORKDIR /work

COPY requirements.txt /work/requirements.txt

USER $USER
RUN pip install --user --upgrade pip && \
    pip install --user jupyter jupyterlab \
    pip install --user -r requirements.txt

EXPOSE 8888
ENTRYPOINT ["jupyter", "lab" "--ip=0.0.0.0"]

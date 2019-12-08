#ベースイメージ
FROM python:3


RUN apt-get update \
    && apt-get install -y git\
    && apt-get install -y make\
    && apt-get install -y curl\
    && apt-get install -y xz-utils\
    && apt-get install -y file\
    && apt-get install -y nodejs\
    && apt-get install -y npm\
    && apt-get install -y sudo\
    && apt-get install -y wget

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

RUN apt-get update\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# delete warnings on docker building.
ENV DEBCONF_NOWARNINGS yes


COPY . ${PWD}/
RUN pip install -r requirement.txt
RUN jupyter labextension install @jupyterlab/plotly-extension
FROM continuumio/anaconda3

COPY ./ /app/

WORKDIR /app

RUN conda -f environment.yml && conda activate liotorch && pip install .
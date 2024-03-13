# create conda env in docker image from env yaml file
FROM continuumio/miniconda3:latest as build

ADD ./environment.yaml /environment.yaml
RUN conda config --set always_yes yes --set changeps1 no && \
    conda config --append channels conda-forge && \
    conda install -c conda-forge mamba conda-pack && \
    mamba env create -f /environment.yaml -n your_env && \
    conda clean -ay && \
    conda-pack -n your_env -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
RUN /venv/bin/conda-unpack

# build runtime environment only
FROM debian:buster AS runtime
#COPY ./entrypoint.sh /entrypoint.sh
#COPY ./src /yipeeo/src
COPY --from=build /venv /venv
#ENTRYPOINT ["/entrypoint.sh"]


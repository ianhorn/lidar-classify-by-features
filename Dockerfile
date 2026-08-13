FROM condaforge/miniforge3:26.3.2-3

WORKDIR /usr/local/app

COPY environment.yml ./

RUN mamba env create -f environment.yml

COPY src ./src
COPY lidar-files ./lidar-files
COPY building-files ./building-files
COPY lidar-features ./lidar-features

SHELL ["mamba", "run", "-n", "lidar-classification", "/bin/bash", "-c"]





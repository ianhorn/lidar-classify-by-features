FROM condaforge/miniforge3:26.3.2-3

WORKDIR /usr/local/app

COPY environment.yml /tmp/environment.yml
# Install the environment into the base environment
RUN mamba install -y -n base -f /tmp/environment.yml && \
    mamba clean --all --yes

# Optional: activate environment for RUN commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN mamba env create -f environment.yml

COPY src ./src
COPY lidar-files ./lidar-files
COPY building-files ./building-files
COPY lidar-features ./lidar-features

# SHELL ["mamba", "run", "-n", "lidar-classification", "/bin/bash", "-c"]

ENTRYPOINT ["mamba", "run", "-n", "lidar-classification", "--no-capture-output"]
CMD ["python", "src/main.py"]


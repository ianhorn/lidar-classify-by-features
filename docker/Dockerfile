# One image, two Batch job definitions: a CPU-only job definition runs
# calculate_point_features.py/backfill_return_number.py, a GPU job
# definition (device_requirements: GPU) runs classify_tiles.py. Using a CUDA
# base for both keeps this to a single image to build/push instead of
# maintaining a matching CPU-only twin -- the CPU job just never requests a
# GPU-backed instance, so the extra CUDA runtime in the image costs disk
# space but nothing at runtime.
#
# The CUDA *runtime* (not the driver) is what needs to be in the image:
# conda-forge's xgboost build is unpinned in docker/environment.yml so its
# __cuda virtual-package detection picks the GPU build against this image's
# CUDA runtime at `micromamba create` time, and classify_tiles.py's
# device="cuda" needs those same libs at run time. The actual NVIDIA driver
# comes from the EC2 host -- AWS Batch's GPU job definitions mount it in
# automatically, the same way ECS/nvidia-docker do.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/lidar-classification/bin:$PATH
# Without this, Python block-buffers stdout when it isn't a real terminal
# (true for a container's log capture) -- a clean exit flushes everything
# printed, but an abrupt crash (a SIGSEGV from PDAL's C++ layer, e.g.) loses
# whatever was still sitting in the buffer. Cost this a wasted debugging
# pass on a run_phase1_batch.py shard that crashed 3 times with no log
# output showing which tile it was on.
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
       | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

COPY docker/environment.yml /tmp/environment.yml
# CONDA_OVERRIDE_CUDA: the docker build has no GPU passthrough, so
# conda-forge's __cuda virtual-package detection (what correctly picks the
# GPU xgboost build on a real GPU machine) would otherwise see no CUDA and
# silently solve to the CPU-only build instead -- verified by building
# without this once and finding xgboost-*-cpu_* in the resolved package
# list. This forces the solver to treat CUDA 12.4 as present, matching the
# base image, without needing a real GPU visible at build time.
RUN CONDA_OVERRIDE_CUDA=12.4 micromamba create -y -n lidar-classification -f /tmp/environment.yml \
    && micromamba clean --all --yes

WORKDIR /app
COPY src/ src/
COPY scripts/ scripts/

# No fixed CMD -- the Batch job definition's command overrides this per job
# type, e.g. ["scripts/classify_tiles.py", "--device", "cuda"] or
# ["scripts/backfill_return_number.py"].
ENTRYPOINT ["python3"]

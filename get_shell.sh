singularity shell \
    --env PYTHONPATH=/app \
    --bind $PWD/:/app/ \
    --pwd /app \
    --nv \
    fmzip_0.0.2.sif
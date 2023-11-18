import os

cache_folder = os.environ.get("YAO_CACHE")
compressed_model_dir = os.path.join(
    cache_folder, "experiments", "fmzip", "compressed_models_new"
)


def render_job():
    pass

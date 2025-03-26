from .core import uss

index = 0


def map_model_to_server(model_name: str):
    global index
    server = uss[index]
    index = (index + 1) % len(uss)

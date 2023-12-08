from packaging.version import parse as parse_version


def compare_transformers_version(version: str = "v4.28.0", op: str = "eq"):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from transformers import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))


def compare_pytorch_version(version: str = "v2.0.0", op: str = "eq"):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from torch import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))

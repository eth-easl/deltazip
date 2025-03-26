from dataclasses import dataclass


@dataclass
class SwapRequest:
    """
    Request for a Swappable model.
    """

    swap_name: str
    swap_int_id: int
    swap_local_path: str

    def __post_init__(self):
        if self.swap_int_id < 1:
            raise ValueError(f"delta_int_id must be > 0, got {self.swap_int_id}")

    def __eq__(self, value: object) -> bool:
        return isinstance(value, SwapRequest) and self.swap_int_id == value.swap_int_id

    def __hash__(self) -> int:
        return self.swap_int_id


def find_swap_model(base_model, requested_model, swap_modules):
    swappable_modules = [x for x in swap_modules if x.name == requested_model]
    if len(swappable_modules) == 0:
        if requested_model == base_model:
            return base_model, base_model
        else:
            print(
                f"Model not found, requested: {requested_model}, available: {[x.name for x in swap_modules] + [base_model]}"
            )
            return None, None
    return swappable_modules[0].name, swappable_modules[0].local_path

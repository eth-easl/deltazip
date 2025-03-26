from dataclasses import dataclass


@dataclass
class DeltaRequest:
    """
    Request for a Delta model.
    """

    delta_name: str
    delta_int_id: int
    delta_local_path: str

    def __post_init__(self):
        if self.delta_int_id < 1:
            raise ValueError(f"delta_int_id must be > 0, got {self.delta_int_id}")

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, DeltaRequest) and self.delta_int_id == value.delta_int_id
        )

    def __hash__(self) -> int:
        return self.delta_int_id

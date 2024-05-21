from dataclasses import dataclass
from typing import Any, Optional, List


@dataclass
class Result:
    """
    Represents the returned value of a function or an error message.
    This is conceptually similar to Rust's `std::result`.
    """

    value: Optional[Any]
    error_details: Optional[List[str]]
    is_ok: bool

    @staticmethod
    def ok(value: Any = None) -> "Result":
        return Result(value, None, is_ok=True)

    @staticmethod
    def err(error_details: List[str]) -> "Result":
        return Result(None, error_details, is_ok=False)

    def __bool__(self) -> bool:
        return self.is_ok

    def __getattribute__(self, name: str) -> Any:
        if name == "value":
            assert self.is_ok, "Cannot retrieve value of an error result"
        if name == "error_details":
            assert not self.is_ok, "Cannot retrieve error details of an ok result"

        return super().__getattribute__(name)

    def __class_getitem__(cls, item):
        return f"{cls.__name__}[{item.__name__}]"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if self.is_ok:
            return f"Result.ok({self.value})"
        return f"Result.err({self.error_details})"

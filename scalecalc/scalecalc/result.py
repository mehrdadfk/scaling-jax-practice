"""Option/Result: Scala-style sealed ADTs for optional and fallible values.

Ports Scala's `Option[A]` (`Some`/`None`) and `Either[E, A]` (here `Result`,
`Ok`/`Err`) to Python. Python's stdlib relies on `None`-checks and exceptions
for these cases instead, which don't compose via `map`/`flat_map` chains and
don't force callers to handle the failure branch. These types do, and also
support structural pattern matching (`match value: case Some(x): ...`) since
each variant is a frozen dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


class Option(Generic[T]):
    """Base class for Some[T] / Nothing. Construct via Some(...), NOTHING, or Option.of(...)."""

    def is_some(self) -> bool:
        return isinstance(self, Some)

    def is_nothing(self) -> bool:
        return isinstance(self, Nothing)

    def map(self, f: Callable[[T], U]) -> "Option[U]":
        return Some(f(self.value)) if isinstance(self, Some) else NOTHING

    def flat_map(self, f: Callable[[T], "Option[U]"]) -> "Option[U]":
        return f(self.value) if isinstance(self, Some) else NOTHING

    def get_or_else(self, default: T) -> T:
        return self.value if isinstance(self, Some) else default

    def unwrap(self) -> T:
        if isinstance(self, Some):
            return self.value
        raise ValueError("called unwrap() on Nothing")

    @staticmethod
    def of(value: T | None) -> "Option[T]":
        """Lift a nullable value: None becomes NOTHING, anything else becomes Some(value)."""
        return NOTHING if value is None else Some(value)


@dataclass(frozen=True)
class Some(Option[T]):
    value: T


@dataclass(frozen=True)
class Nothing(Option):
    pass


NOTHING: Nothing = Nothing()


class Result(Generic[T, E]):
    """Base class for Ok[T, E] / Err[T, E]. Mirrors Scala's Either, biased to the Ok case."""

    def is_ok(self) -> bool:
        return isinstance(self, Ok)

    def is_err(self) -> bool:
        return isinstance(self, Err)

    def map(self, f: Callable[[T], U]) -> "Result[U, E]":
        return Ok(f(self.value)) if isinstance(self, Ok) else self

    def map_err(self, f: Callable[[E], F]) -> "Result[T, F]":
        return Err(f(self.error)) if isinstance(self, Err) else self

    def flat_map(self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        return f(self.value) if isinstance(self, Ok) else self

    def get_or_else(self, default: T) -> T:
        return self.value if isinstance(self, Ok) else default

    def unwrap(self) -> T:
        if isinstance(self, Ok):
            return self.value
        raise ValueError(f"called unwrap() on Err: {self.error!r}")

    @staticmethod
    def try_(f: Callable[..., T], *args, **kwargs) -> "Result[T, Exception]":
        """Run f and capture any raised exception as an Err instead of propagating it."""
        try:
            return Ok(f(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - intentionally broad, mirrors Scala's Try
            return Err(exc)


@dataclass(frozen=True)
class Ok(Result[T, E]):
    value: T


@dataclass(frozen=True)
class Err(Result[T, E]):
    error: E

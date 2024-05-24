"""Classes for data containers."""

from collections.abc import Sequence
from typing import TypeAlias

IlocIndexer: TypeAlias = int | slice | Sequence[int] | Sequence[bool]

"""In-memory cache for LFD map computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class LFDCache:
    """Simple in-memory cache keyed by (image_id, params_hash)."""

    store: Dict[Tuple[str, int], Any]

    def __init__(self) -> None:
        self.store = {}

    def get(self, image_id: str, params_hash: int) -> Optional[Any]:
        return self.store.get((image_id, params_hash))

    def set(self, image_id: str, params_hash: int, value: Any) -> None:
        self.store[(image_id, params_hash)] = value

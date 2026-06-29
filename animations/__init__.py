"""Animation helpers for reviewing synchronized tracking and event data."""

from .adapters import (
    build_event_animation,
    prepare_event_animation_data,
    rank_sync_coordinate_deltas,
    save_animation,
    save_event_animation,
)
from .animator import Animator, MarkerSpec, VectorSpec

__all__ = [
    "Animator",
    "MarkerSpec",
    "VectorSpec",
    "build_event_animation",
    "prepare_event_animation_data",
    "rank_sync_coordinate_deltas",
    "save_animation",
    "save_event_animation",
]

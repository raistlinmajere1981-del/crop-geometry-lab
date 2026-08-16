"""Manual override ledger used by API clients and tests.

The browser editor keeps the resolved circle set, while this module defines the
same precedence rule in reusable Python: a manual delete, update, or addition
always wins over current or subsequently refreshed automatic candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

from .feature_table import CircleFeature

EditKind = Literal["add", "update", "delete"]


@dataclass(frozen=True, slots=True)
class ManualEdit:
    kind: EditKind
    circle_id: str
    circle: CircleFeature | None = None

    def __post_init__(self) -> None:
        if self.kind in {"add", "update"} and self.circle is None:
            raise ValueError(f"{self.kind} edits require a circle")
        if self.circle is not None and self.circle.id != self.circle_id:
            raise ValueError("circle_id must match circle.id")


class ManualOverrideLedger:
    """Resolve automatic candidates under a persistent sequence of user edits."""

    def __init__(self, automatic: Iterable[CircleFeature] = ()) -> None:
        self._automatic = {circle.id: circle for circle in automatic}
        self._edits: dict[str, ManualEdit] = {}

    def replace_automatic(self, circles: Iterable[CircleFeature]) -> None:
        """Refresh detection results without discarding any user decision."""

        self._automatic = {circle.id: circle for circle in circles}

    def add(self, circle: CircleFeature) -> None:
        manual = replace(circle, source="manual")
        self._edits[circle.id] = ManualEdit("add", circle.id, manual)

    def update(self, circle: CircleFeature) -> None:
        manual = replace(circle, source="manual_override")
        self._edits[circle.id] = ManualEdit("update", circle.id, manual)

    def delete(self, circle_id: str) -> None:
        self._edits[circle_id] = ManualEdit("delete", circle_id)

    def resolve(self) -> list[CircleFeature]:
        resolved = dict(self._automatic)
        for circle_id, edit in self._edits.items():
            if edit.kind == "delete":
                resolved.pop(circle_id, None)
            elif edit.circle is not None:
                resolved[circle_id] = edit.circle
        return sorted(resolved.values(), key=lambda circle: circle.id)

    @property
    def edits(self) -> tuple[ManualEdit, ...]:
        return tuple(self._edits.values())


def apply_manual_edits(
    automatic: Iterable[CircleFeature], edits: Iterable[ManualEdit]
) -> list[CircleFeature]:
    ledger = ManualOverrideLedger(automatic)
    for edit in edits:
        if edit.kind == "delete":
            ledger.delete(edit.circle_id)
        elif edit.kind == "add" and edit.circle is not None:
            ledger.add(edit.circle)
        elif edit.kind == "update" and edit.circle is not None:
            ledger.update(edit.circle)
    return ledger.resolve()


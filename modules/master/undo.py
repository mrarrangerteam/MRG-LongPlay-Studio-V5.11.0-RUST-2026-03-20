"""
CommandHistory — Undo/Redo system for mastering parameter changes.
"""

from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class Command:
    """A single parameter change command."""
    module: str      # e.g., "maximizer", "equalizer", "dynamics", "imager"
    param: str       # e.g., "gain_db", "threshold", "width"
    old_val: Any
    new_val: Any
    description: str = ""


class CommandHistory:
    """
    Undo/Redo stack with max 50 entries.

    Usage:
        history = CommandHistory()
        history.push(Command("maximizer", "gain_db", 0.0, 5.0, "Gain +5dB"))
        cmd = history.undo()  # returns Command to reverse
        cmd = history.redo()  # returns Command to re-apply
    """

    MAX_SIZE = 50

    def __init__(self):
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    def push(self, cmd: Command):
        """Record a new command. Clears redo stack."""
        self._undo_stack.append(cmd)
        if len(self._undo_stack) > self.MAX_SIZE:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> Optional[Command]:
        """Pop last command from undo stack, push to redo. Returns command to reverse."""
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        self._redo_stack.append(cmd)
        return cmd

    def redo(self) -> Optional[Command]:
        """Pop last command from redo stack, push to undo. Returns command to re-apply."""
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        self._undo_stack.append(cmd)
        return cmd

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def last_undo_description(self) -> str:
        if self._undo_stack:
            return self._undo_stack[-1].description
        return ""

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

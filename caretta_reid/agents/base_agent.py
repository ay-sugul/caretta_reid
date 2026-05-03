"""Abstract base classes for ReID agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Defines the contract for all agents in the pipeline."""

    @abstractmethod
    def execute(self, message: InputT) -> OutputT:
        """Processes an input message and returns the next typed message.

        Args:
            message: Typed input payload for the agent.

        Returns:
            Typed output payload produced by the agent.
        """
        raise NotImplementedError

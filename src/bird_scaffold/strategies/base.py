from __future__ import annotations

from abc import ABC, abstractmethod

from bird_scaffold.llm_client import OpenAICompatibleText2SQLClient
from bird_scaffold.types import BirdExample, DatabaseContext, GenerationResult


class SQLGenerationStrategy(ABC):
    name = "base"
    requires_llm = True

    @abstractmethod
    def generate(
        self,
        example: BirdExample,
        db_context: DatabaseContext,
        llm_client: OpenAICompatibleText2SQLClient | None,
        include_evidence: bool,
    ) -> GenerationResult:
        raise NotImplementedError

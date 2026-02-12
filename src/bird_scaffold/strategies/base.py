from __future__ import annotations

from abc import ABC, abstractmethod

from bird_scaffold.openai_client import OpenAIText2SQLClient
from bird_scaffold.types import BirdExample, DatabaseContext, GenerationResult


class SQLGenerationStrategy(ABC):
    name = "base"
    requires_openai = True

    @abstractmethod
    def generate(
        self,
        example: BirdExample,
        db_context: DatabaseContext,
        openai_client: OpenAIText2SQLClient | None,
        include_evidence: bool,
    ) -> GenerationResult:
        raise NotImplementedError

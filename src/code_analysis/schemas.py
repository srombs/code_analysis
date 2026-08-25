"""Schemas that define the data exchanged by the code-analysis agent."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Finding(BaseModel):
    """One actionable issue found in a source file."""

    model_config = ConfigDict(extra="forbid")

    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)
    problem: str
    solution: str
    severity: Literal["low", "medium", "high"]
    start_character: int | None = Field(default=None, ge=0)
    end_character: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_positions(self) -> "Finding":
        """Ensure every reported source range runs forward."""
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        if (
            self.start_character is not None
            and self.end_character is not None
            and self.end_character < self.start_character
        ):
            raise ValueError("end_character must be greater than or equal to start_character")
        return self


class AnalysisResult(BaseModel):
    """The validated, application-specific result of a code analysis."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    findings: list[Finding]

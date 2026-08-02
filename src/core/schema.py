"""
Schema definition for the Docstring Generator dataset.

Each dataset row is an instruction-following triplet:
    instruction: fixed task description
    input:       a Python function definition (as source text)
    output:      a Google-style docstring for that function
"""

from pydantic import BaseModel, field_validator

ALLOWED_INSTRUCTION = "Generate a Google-style docstring for this Python function"


class DocstringExample(BaseModel):
    instruction: str
    input: str
    output: str

    @field_validator("instruction")
    @classmethod
    def instruction_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("instruction cannot be empty")
        return v

    @field_validator("input")
    @classmethod
    def input_contains_function(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("input cannot be empty")
        if "def " not in v:
            raise ValueError("input must contain a Python function definition (def ...)")
        return v

    @field_validator("output")
    @classmethod
    def output_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("output (docstring) cannot be empty")
        return v

    def is_google_style(self) -> bool:
        """Heuristic check for Google-style section headers."""
        google_sections = ("Args:", "Returns:", "Raises:", "Yields:", "Example:", "Examples:")
        return any(section in self.output for section in google_sections)

"""
OpenAI Responses API client for AlphaLens.

This module is responsible only for LLM text generation. It does not fetch
financial data, calculate metrics, build evidence packages, or manage Streamlit
UI state.

AlphaLens uses deterministic financial evidence assembled in Python, then asks
OpenAI to synthesize that evidence into balanced research-style analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI, OpenAIError

from src.config import settings


class LLMClientError(Exception):
    """
    Base exception for AlphaLens LLM client errors.
    """

    pass


class LLMAuthenticationError(LLMClientError):
    """
    Raised when the OpenAI API key is missing.
    """

    pass


class LLMResponseError(LLMClientError):
    """
    Raised when OpenAI returns an unusable response.
    """

    pass


@dataclass
class LLMGenerationConfig:
    """
    Configuration for OpenAI text generation.

    Attributes:
        model: OpenAI model name.
        max_output_tokens: Maximum number of generated output tokens.
        temperature: Sampling temperature. Lower values produce more stable,
            less varied output.
    """

    model: str = settings.openai_model
    max_output_tokens: int = 900
    temperature: float = 0.2


class AlphaLensLLMClient:
    """
    Thin wrapper around the OpenAI Responses API.

    Args:
        api_key: OpenAI API key.
        generation_config: Text generation settings.
        client: Optional OpenAI client, useful for testing.
    """

    def __init__(
        self,
        api_key: str | None = settings.openai_api_key,
        generation_config: LLMGenerationConfig | None = None,
        client: OpenAI | None = None,
    ) -> None:
        if not api_key:
            raise LLMAuthenticationError(
                "OPENAI_API_KEY is missing. Add it to your .env file."
            )

        self.generation_config = generation_config or LLMGenerationConfig()
        self.client = client or OpenAI(api_key=api_key)

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Generate text using the OpenAI Responses API.

        Args:
            system_prompt: System-level behavior instructions.
            user_prompt: User prompt containing analysis instructions and evidence.

        Returns:
            Generated text from the model.

        Raises:
            LLMClientError: If the API call fails or output is unusable.
        """

        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty.")

        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty.")

        try:
            response = self.client.responses.create(
                model=self.generation_config.model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=self.generation_config.max_output_tokens,
                temperature=self.generation_config.temperature,
            )
        except OpenAIError as exc:
            raise LLMClientError(f"OpenAI Responses API call failed: {exc}") from exc
        except Exception as exc:
            raise LLMClientError(f"Unexpected LLM client error: {exc}") from exc

        return extract_response_text(response)

    def generate_from_prompt_payload(
        self,
        prompt_payload: dict[str, Any],
    ) -> str:
        """
        Generate text from a prompt payload produced by src.prompts.py.

        Args:
            prompt_payload: Dictionary returned by build_prompt_payload.

        Returns:
            Generated text from the model.
        """

        system_prompt = prompt_payload.get("system_prompt")
        user_prompt = prompt_payload.get("user_prompt")

        if not system_prompt:
            raise ValueError("prompt_payload must include 'system_prompt'.")

        if not user_prompt:
            raise ValueError("prompt_payload must include 'user_prompt'.")

        return self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


def extract_response_text(response: Any) -> str:
    """
    Extract text from an OpenAI Responses API response.

    The OpenAI SDK commonly exposes generated text through response.output_text.
    This function also includes a defensive fallback that walks the response
    output structure in case output_text is unavailable or empty.

    Args:
        response: Raw response object from client.responses.create.

    Returns:
        Generated text.

    Raises:
        LLMResponseError: If no text can be extracted.
    """

    output_text = getattr(response, "output_text", None)

    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    fallback_text = extract_text_from_output_items(response)

    if fallback_text.strip():
        return fallback_text.strip()

    raise LLMResponseError(
        "OpenAI response did not contain usable text output."
    )


def extract_text_from_output_items(response: Any) -> str:
    """
    Defensive fallback for extracting text from response.output.

    This is intentionally tolerant because SDK response objects may expose
    attributes, dictionaries, or nested content blocks depending on version.

    Args:
        response: Raw OpenAI response object.

    Returns:
        Concatenated text content, or an empty string.
    """

    output_items = getattr(response, "output", None)

    if not output_items:
        return ""

    text_parts: list[str] = []

    for item in output_items:
        content = get_attr_or_key(item, "content")

        if not content:
            continue

        for content_block in content:
            block_type = get_attr_or_key(content_block, "type")
            block_text = get_attr_or_key(content_block, "text")

            if block_type in {"output_text", "text"} and isinstance(block_text, str):
                text_parts.append(block_text)

    return "\n".join(text_parts)


def get_attr_or_key(item: Any, key: str) -> Any:
    """
    Read a value from either an object attribute or dictionary key.

    Args:
        item: Object or dictionary.
        key: Attribute/key name.

    Returns:
        Value if found, otherwise None.
    """

    if isinstance(item, dict):
        return item.get(key)

    return getattr(item, key, None)


def create_llm_client() -> AlphaLensLLMClient:
    """
    Factory function for creating the configured AlphaLens LLM client.

    Returns:
        AlphaLensLLMClient instance.
    """

    return AlphaLensLLMClient()


def generate_analysis_text(prompt_payload: dict[str, Any]) -> str:
    """
    Convenience function for generating analysis text from a prompt payload.

    Args:
        prompt_payload: Dictionary returned by src.prompts.build_prompt_payload.

    Returns:
        Generated research-style analysis text.
    """

    client = create_llm_client()

    return client.generate_from_prompt_payload(prompt_payload)
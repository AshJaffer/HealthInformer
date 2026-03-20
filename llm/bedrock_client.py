"""AWS Bedrock client for Claude 3.5 Sonnet.

Uses boto3 invoke_model with the Anthropic Messages API format.
Requires AWS credentials (access key, secret, session token) in .env
or via the AWS credential chain.

Usage:
    from llm.bedrock_client import BedrockLLM

    llm = BedrockLLM()
    answer = llm.generate("Explain type 2 diabetes.", system_prompt="Be concise.")
"""

import json

import boto3

from config.settings import (
    AWS_REGION,
    BEDROCK_MAX_TOKENS,
    BEDROCK_MODEL_ID,
    BEDROCK_TEMPERATURE,
)
from llm.base import BaseLLM


class BedrockLLM(BaseLLM):
    """AWS Bedrock client using the Anthropic Messages API."""

    def __init__(
        self,
        model_id: str = BEDROCK_MODEL_ID,
        region: str = AWS_REGION,
    ) -> None:
        """Initialize the Bedrock runtime client.

        Args:
            model_id: Bedrock model identifier.
            region: AWS region for the Bedrock endpoint.
        """
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def generate(
        self,
        prompt: str,
        max_tokens: int = BEDROCK_MAX_TOKENS,
        temperature: float = BEDROCK_TEMPERATURE,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response via Bedrock invoke_model.

        Uses the Anthropic Messages API format expected by Claude models.

        Args:
            prompt: User message content.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.
            system_prompt: Optional system message.

        Returns:
            Generated text string.
        """
        body: dict = {
            "anthropic_version": "bedrock-2023-10-16",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        if system_prompt:
            body["system"] = system_prompt

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]


if __name__ == "__main__":
    llm = BedrockLLM()
    result = llm.generate("What is type 2 diabetes? Answer in one sentence.")
    print(result)

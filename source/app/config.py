"""Shared runtime configuration and per-agent model helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class AgentModelConfig(BaseModel):
    """Per-agent model configuration."""

    model_name: str = Field(default="", description="Chat model name for the agent.")
    api_key: str = Field(default="", description="API key for the agent.")
    base_url: Optional[str] = Field(default=None, description="Optional OpenAI-compatible base URL.")


class RuntimeConfig(BaseModel):
    """Runtime configuration shared across stages."""

    max_build_retry: int = Field(default=2, description="Build-stage retry limit.")
    max_poc_retry: int = Field(default=2, description="PoC-stage retry limit.")
    workspace_root: str = Field(default="workspaces", description="Workspace root directory.")
    knowledge_max_reference_depth: int = Field(default=3, description="Maximum recursive URL depth for the knowledge stage.")
    knowledge_max_fetch_count: int = Field(default=12, description="Maximum number of fetched references for one knowledge-stage run.")
    knowledge_max_selected_references: int = Field(default=24, description="Maximum number of selected references retained after prioritization.")
    knowledge_max_discovered_references_per_page: int = Field(default=8, description="Maximum number of recursively discovered references kept from one page.")
    knowledge_max_output_references: int = Field(default=30, description="Maximum number of references written to final knowledge output.")
    knowledge_fetch_timeout_seconds: int = Field(default=8, description="Timeout for each remote fetch in the knowledge stage.")
    knowledge_enable_llm_curation: bool = Field(default=False, description="Whether the knowledge stage should invoke the curation LLM.")
    llm_timeout_seconds: int = Field(default=30, description="Timeout passed to LLM client requests.")
    build_agent_timeout_seconds: int = Field(default=60, description="Timeout for build-agent LLM requests.")
    poc_agent_timeout_seconds: int = Field(default=90, description="Timeout for poc-agent LLM requests.")


class AppConfig(BaseModel):
    """Application configuration snapshot."""

    knowledge_agent: AgentModelConfig = Field(default_factory=AgentModelConfig)
    build_agent: AgentModelConfig = Field(default_factory=AgentModelConfig)
    poc_agent: AgentModelConfig = Field(default_factory=AgentModelConfig)
    verify_agent: AgentModelConfig = Field(default_factory=AgentModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def _load_agent_config(prefix: str, default_model: str = "gpt-4.1-mini") -> AgentModelConfig:
    """Load one agent's model configuration from `.env`."""

    return AgentModelConfig(
        model_name=os.getenv(f"{prefix}_MODEL", default_model),
        api_key=os.getenv(f"{prefix}_API_KEY", ""),
        base_url=os.getenv(f"{prefix}_BASE_URL") or None,
    )


@lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    """Load environment variables from `.env` and build a config object."""

    load_dotenv(override=True)
    return AppConfig(
        knowledge_agent=_load_agent_config("KNOWLEDGE_AGENT"),
        build_agent=_load_agent_config("BUILD_AGENT"),
        poc_agent=_load_agent_config("POC_AGENT"),
        verify_agent=_load_agent_config("VERIFY_AGENT"),
        runtime=RuntimeConfig(
            max_build_retry=int(os.getenv("MAX_BUILD_RETRY", "2")),
            max_poc_retry=int(os.getenv("MAX_POC_RETRY", "2")),
            workspace_root=os.getenv("WORKSPACE_ROOT", "workspaces"),
            knowledge_max_reference_depth=int(os.getenv("KNOWLEDGE_MAX_REFERENCE_DEPTH", "3")),
            knowledge_max_fetch_count=int(os.getenv("KNOWLEDGE_MAX_FETCH_COUNT", "12")),
            knowledge_max_selected_references=int(os.getenv("KNOWLEDGE_MAX_SELECTED_REFERENCES", "24")),
            knowledge_max_discovered_references_per_page=int(os.getenv("KNOWLEDGE_MAX_DISCOVERED_REFERENCES_PER_PAGE", "8")),
            knowledge_max_output_references=int(os.getenv("KNOWLEDGE_MAX_OUTPUT_REFERENCES", "30")),
            knowledge_fetch_timeout_seconds=int(os.getenv("KNOWLEDGE_FETCH_TIMEOUT_SECONDS", "8")),
            knowledge_enable_llm_curation=os.getenv("KNOWLEDGE_ENABLE_LLM_CURATION", "0").strip().lower() in {"1", "true", "yes", "on"},
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            build_agent_timeout_seconds=int(os.getenv("BUILD_AGENT_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "60"))),
            poc_agent_timeout_seconds=int(os.getenv("POC_AGENT_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "90"))),
        ),
    )


def get_agent_model_config(agent_name: str) -> AgentModelConfig:
    """Return the model config for a named agent."""

    config = load_app_config()
    try:
        return getattr(config, agent_name)
    except AttributeError as error:
        raise ValueError(f"Unknown agent name: {agent_name}") from error


def build_chat_model(
    agent_name: str,
    model_name: Optional[str] = None,
    temperature: float = 0,
    timeout_seconds: Optional[int] = None,
):
    """Create a LangChain chat model for a specific agent."""

    from langchain_openai import ChatOpenAI

    agent_config = get_agent_model_config(agent_name)
    runtime = load_app_config().runtime
    if not agent_config.api_key:
        raise RuntimeError(f"{agent_name} API key is missing in .env.")

    return ChatOpenAI(
        model=model_name or agent_config.model_name,
        temperature=temperature,
        api_key=agent_config.api_key,
        base_url=agent_config.base_url,
        timeout=timeout_seconds or runtime.llm_timeout_seconds,
    )

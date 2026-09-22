"""Typed access to everything configured in .env.

One settings object, imported by the producer, the streaming job, the API, the
training scripts and the analyst app. Putting it in one place means a variable
is named once and validated once: if .env.example and the code drift apart,
``tests/test_config.py`` fails rather than something breaking at 2am in a demo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# The providers the assistant knows how to talk to. Validated at startup so a
# typo in .env is a clear error rather than a confusing fallback.
LLM_PROVIDERS = frozenset({"ollama", "azure_openai"})
EMBEDDING_PROVIDERS = frozenset({"sentence_transformers", "azure_openai"})


class Settings(BaseSettings):
    """Platform configuration, read from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        # Unknown keys in .env are ignored rather than fatal: the file is shared
        # with docker compose, which has its own variables.
        extra="ignore",
        case_sensitive=False,
    )

    # --- Kafka ---
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_bootstrap_servers_host: str = "localhost:29092"
    kafka_topic_payments: str = "payments"
    kafka_topic_decisions: str = "decisions"

    # --- Producer ---
    # TransactionDT is a seconds offset; this is the date it is mapped onto.
    synthetic_epoch: str = "2023-01-01T00:00:00Z"
    replay_speedup: float = 3600.0
    producer_max_events: int = 0

    # --- Redis ---
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_feature_ttl_seconds: int = 172_800

    # --- Storage ---
    delta_bronze_path: str = "data/delta/bronze"
    delta_silver_path: str = "data/delta/silver"
    delta_gold_path: str = "data/delta/gold"
    delta_decisions_path: str = "data/delta/decisions"
    # Labels arrive late, so they live in their own table rather than being
    # glued onto the payment data (see training/chargebacks.py).
    delta_chargebacks_path: str = "data/delta/chargebacks"
    delta_training_path: str = "data/delta/training"
    duckdb_path: str = "data/warehouse/fraud.duckdb"

    # --- PII ---
    pii_hash_salt: str = "change-me-generate-a-random-64-char-hex-string"

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "fraud-xgboost"
    mlflow_registered_model: str = "fraud-xgboost"
    mlflow_champion_alias: str = "champion"

    # --- Serving ---
    scoring_api_url: str = "http://api:8000"
    scoring_api_url_host: str = "http://localhost:8000"
    threshold_review: float = Field(default=0.30, ge=0.0, le=1.0)
    threshold_block: float = Field(default=0.80, ge=0.0, le=1.0)
    cost_false_block: float = 25.0
    cost_review: float = 5.0

    # --- GenAI: which provider serves the assistant ---
    # The default is local and free, so the repository runs for anyone who
    # clones it. Azure OpenAI is opt-in because it bills per token and needs a
    # subscription; see docs/design_decisions.md for why both exist.
    llm_provider: str = "ollama"
    embedding_provider: str = "sentence_transformers"

    # --- GenAI: local provider (Ollama + sentence-transformers) ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- GenAI: Azure OpenAI ---
    # Endpoint looks like https://<resource>.openai.azure.com - no trailing path.
    azure_openai_endpoint: str = ""
    # SecretStr so the key cannot leak through a settings repr in a log line or
    # a Streamlit exception traceback. Read it with .get_secret_value().
    azure_openai_api_key: SecretStr = SecretStr("")
    # Pinned to a GA version rather than a preview: preview versions are removed
    # on a schedule, and a portfolio project that stops working is worse than
    # one using a slightly older API.
    azure_openai_api_version: str = "2024-10-21"
    # These are Azure *deployment* names, which you choose in the portal. They
    # often differ from the underlying model name, which is the single most
    # common cause of a 404 from Azure OpenAI.
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # --- GenAI: past-case store ---
    pgvector_host: str = "pgvector"
    pgvector_port: int = 5432
    pgvector_db: str = "fraud_cases"
    pgvector_user: str = "fraud"
    pgvector_password: str = "fraud_local_dev_only"
    rag_top_k: int = 5

    # --- Labels ---
    chargeback_min_delay_days: int = 7
    chargeback_max_delay_days: int = 60
    label_maturity_days: int = 60

    # --- Reproducibility ---
    random_seed: int = 42

    @model_validator(mode="after")
    def _check_provider_configuration(self) -> Settings:
        """Fail at startup rather than at the first analyst request.

        A missing endpoint or key is a configuration mistake, and the place to
        find out is the moment the process starts - not thirty seconds into a
        demo when an analyst opens the first flagged payment and the assistant
        silently falls back to showing no summary.
        """
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(LLM_PROVIDERS)}")
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {sorted(EMBEDDING_PROVIDERS)}")

        uses_azure = "azure_openai" in {self.llm_provider, self.embedding_provider}
        if uses_azure:
            if not self.azure_openai_endpoint:
                raise ValueError(
                    "AZURE_OPENAI_ENDPOINT is required when a provider is azure_openai"
                )
            if not self.azure_openai_api_key.get_secret_value():
                raise ValueError("AZURE_OPENAI_API_KEY is required when a provider is azure_openai")
        return self

    @property
    def uses_paid_provider(self) -> bool:
        """True when a run will bill somebody.

        Surfaced in the analyst app and the eval report so a cost is never a
        surprise, and so a published metric records which provider produced it.
        """
        return "azure_openai" in {self.llm_provider, self.embedding_provider}

    def path(self, relative: str) -> Path:
        """Resolve a configured relative path against the repository root."""
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate

    @property
    def uses_default_pii_salt(self) -> bool:
        """True when the placeholder salt from .env.example is still in use.

        Tokenisation with a publicly known salt is reversible by anyone with the
        same code, so the pipeline warns loudly rather than silently pretending
        the data is masked.
        """
        return self.pii_hash_salt.startswith("change-me")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object (parsed once)."""
    return Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # Mistral
    mistral_api_key: str = Field(..., description='Mistral API key')
    mistral_small_model: str = Field(default='mistral-small-latest')
    mistral_large_model: str = Field(default='mistral-large-latest')
    mistral_timeout: int = Field(default=60)
    mistral_max_retries: int = Field(default=3)

    # Neo4j
    neo4j_uri: str = Field(...)
    neo4j_username: str = Field(...)
    neo4j_password: str = Field(...)
    neo4j_database: str = Field(default='neo4j')

    # Pinecone
    pinecone_api_key: str = Field(...)
    pinecone_index_name: str = Field(default='graphrag-research')
    pinecone_environment: str = Field(default='us-east-1')

    # Embeddings — now via Mistral (mistral-embed model)
    embedding_model: str = Field(default='mistral-embed')
    embedding_dim: int = Field(default=1024)
    embedding_batch_size: int = Field(default=32)

    # HuggingFace — kept as optional so old .env files don't break
    huggingface_api_token: str = Field(default='')
    huggingface_api_url: str = Field(default='')
    huggingface_timeout: int = Field(default=30)
    huggingface_max_retries: int = Field(default=3)

    # App
    app_host: str = Field(default='0.0.0.0')
    app_port: int = Field(default=8000)
    max_upload_size_mb: int = Field(default=50)
    cors_origins: list[str] = Field(
        default=['http://localhost:5173', 'http://localhost:3000']
    )
    job_ttl_seconds: int = Field(default=86400)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
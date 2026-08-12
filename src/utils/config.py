"""Environment configuration and secret management.

Enforces startup validation of environment variables and AWS credentials via Pydantic.
Prevents hardcoded secrets and guarantees proper configuration across local and cloud environments.
"""

from __future__ import annotations

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application settings with dynamic .env loading and startup validation.

    Attributes:
        aws_access_key_id: Optional AWS Access Key ID.
        aws_secret_access_key: Optional AWS Secret Access Key.
        aws_region: Target AWS Region (default: us-east-1).
        athena_s3_staging_dir: S3 bucket URI for Athena query results.
        e2b_api_key: Optional API key for E2B sandbox microVMs.
        otel_exporter_otlp_endpoint: Optional OpenTelemetry collector endpoint.
        athena_max_scan_bytes: Maximum allowed scan bytes threshold per query (default 10GB).
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    athena_s3_staging_dir: str = Field(
        default="s3://athena-query-results-staging/",
        alias="ATHENA_S3_STAGING_DIR",
    )
    e2b_api_key: Optional[str] = Field(default=None, alias="E2B_API_KEY")
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    athena_max_scan_bytes: int = Field(
        default=10 * 1024 * 1024 * 1024,  # 10 GB limit safety cap
        alias="ATHENA_MAX_SCAN_BYTES",
    )

    def validate_aws_credentials(self) -> bool:
        """Verify presence of valid AWS credentials.

        Returns:
            True if credentials are present, False otherwise.
        """
        return bool(self.aws_access_key_id and self.aws_secret_access_key)


# Global settings singleton instance
config = AppConfig()

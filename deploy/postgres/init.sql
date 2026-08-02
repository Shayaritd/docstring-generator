-- Minimal request-logging schema for the optional Postgres logging profile.
-- The API does not currently write to this table (no logging-to-Postgres
-- code exists in api/app/main.py) — this schema is provided so the table
-- structure is ready if/when that integration is added. Until then, the
-- `postgres` service will start and be empty; logs remain in stdout/JSON
-- (see api/app/logging_config.py) as the actual current logging path.

CREATE TABLE IF NOT EXISTS request_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID DEFAULT gen_random_uuid(),
    endpoint TEXT NOT NULL,
    function_code TEXT,
    docstring_generated TEXT,
    model_label TEXT,
    latency_ms DOUBLE PRECISION,
    status_code INTEGER,
    error_detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_endpoint ON request_logs (endpoint);

-- Table 1: Run metadata
CREATE TABLE IF NOT EXISTS sweep_run (
    run_id SERIAL PRIMARY KEY,
    github_pipeline_id BIGINT UNIQUE,
    run_contents TEXT NOT NULL,        -- 'lead models', 'nightly', etc.
    card_type TEXT NOT NULL,           -- 'wormhole_b0', 'blackhole'
    git_sha TEXT,
    git_branch TEXT,
    run_start_ts TIMESTAMPTZ NOT NULL,
    test_count INT,
    pass_count INT,
    fail_count INT
);

CREATE INDEX IF NOT EXISTS sweep_run_lookup_idx
    ON sweep_run (run_contents, card_type, run_start_ts DESC);

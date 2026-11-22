-- Table 2: Test results (denormalized - includes perf metric inline)
CREATE TABLE IF NOT EXISTS sweep_test (
    test_id SERIAL PRIMARY KEY,
    run_id INT REFERENCES sweep_run(run_id) ON DELETE CASCADE,
    full_test_name TEXT NOT NULL,
    input_hash TEXT,                   -- Hash of test input parameters for cross-run matching
    op_name TEXT,
    model_name TEXT,
    status TEXT NOT NULL,              -- 'pass', 'fail_assert_exception', etc.
    device_fw_duration_ns FLOAT    -- NULL if not measured
);

CREATE INDEX IF NOT EXISTS sweep_test_run_idx ON sweep_test (run_id);
CREATE INDEX IF NOT EXISTS sweep_test_status_idx ON sweep_test (run_id, status);
CREATE INDEX IF NOT EXISTS sweep_test_match_idx ON sweep_test (full_test_name, input_hash);

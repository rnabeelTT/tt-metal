CREATE TABLE ttnn_ops_v6.trace_run (
    trace_run_id    SERIAL PRIMARY KEY,
    trace_uid       TEXT NOT NULL UNIQUE,
    hardware_id     INTEGER
                        REFERENCES ttnn_ops_v6.ttnn_hardware(ttnn_hardware_id),
    tt_metal_sha    TEXT,
    traced_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    config_count    INTEGER,    -- denormalized, updated after aggregate refresh
    notes           TEXT,
    -- The pytest CLI args (everything after `--`) used for this trace
    -- invocation. Lets us answer "have I traced model X with these args
    -- on hardware Y?" without re-running the tracer.
    pytest_args     TEXT,
    -- Trace host software provenance captured from tt-smi -s at trace time.
    tt_kmd          TEXT,
    tt_smi          TEXT,
    tt_firmware     TEXT
);
CREATE INDEX trace_run_trace_uid_idx      ON ttnn_ops_v6.trace_run(trace_uid);
CREATE INDEX trace_run_hardware_id_idx  ON ttnn_ops_v6.trace_run(hardware_id);
CREATE INDEX trace_run_traced_at_idx    ON ttnn_ops_v6.trace_run(traced_at DESC);
CREATE INDEX trace_run_sha_idx          ON ttnn_ops_v6.trace_run(tt_metal_sha)
    WHERE tt_metal_sha IS NOT NULL;
CREATE INDEX trace_run_pytest_args_idx  ON ttnn_ops_v6.trace_run(pytest_args)
    WHERE pytest_args IS NOT NULL;
CREATE INDEX trace_run_tt_kmd_idx  ON ttnn_ops_v6.trace_run(tt_kmd)
    WHERE tt_kmd IS NOT NULL;
CREATE INDEX trace_run_tt_smi_idx  ON ttnn_ops_v6.trace_run(tt_smi)
    WHERE tt_smi IS NOT NULL;
CREATE INDEX trace_run_tt_firmware_idx  ON ttnn_ops_v6.trace_run(tt_firmware)
    WHERE tt_firmware IS NOT NULL;

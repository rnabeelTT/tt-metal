Second bucketed 48-layer run (widths 2..32, device-side GDN slot write, decode-input contract v1; sampling suite only:
the runner stops after a failed suite). 72 passed / 4 skipped / 1 FAILED: `test_request_isolation::test_mixed_params_batch`
(a seeded temperature-0.5 request reproduced in run 1 but differed at its 4th token in run 2: the shuffled admission order
gave it a different decode-width history, and per-row numerics differ between widths). Led to the seeded-full-width rule
(README section 3). Not the final evidence run.

"""Requeue for the 81,920-token budget (user, 2026-11-02: the 32k cap was ours, not the cards'; halve the hard sets to keep ~2 h).
argv[1] = 1 if the Ornith-1.5-35B fix release passed (its HLE / GPQA reruns are queued only then)."""
import fcntl, re, sys
orn_ok = sys.argv[1] == "1"
with open("evalq.txt", "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    L = f.read().split("\n")
    L = [l for l in L if l != "orn text2 hle SAMPLE_N=128 MAX_TOKENS=32768 CONC=32"]  # orn_fix_chain's 32k requeue
    out = []
    for l in L:
        if l.strip() and not l.startswith("#"):
            l = l.replace("MAX_TOKENS=32768", "MAX_TOKENS=81920")
            if l.startswith("q122 lmeval r1_gpqa_diamond"): l = l.replace("MAX_TOKENS", "SAMPLE_N=99 TOTAL=198 MAX_TOKENS")
            if l.startswith("q122 lmeval livecodebench_v6"): l = l.replace("MAX_TOKENS", "SAMPLE_N=88 TOTAL=175 MAX_TOKENS")
            if l.startswith("q122 text2 ojbench"): l = l.replace("SAMPLE_N=64", "SAMPLE_N=32")
            if l.startswith("q122 "): l = "#HOLD-Q122 " + l  # released when the orn reruns start (budget_unhold)
        out.append(l)
    head = ["# 81,920-token budget (user, 2026-11-02): 32k answers were our cap, not the cards'; hard sets halved to stay ~2 h",
            "q122 extend hle BUDGET=81920 SUBSAMPLE=64 CONC=32 TAG=b81920_n64",
            "o122 extend hle BUDGET=81920 SRC_TAG=pp SUBSAMPLE=64 CONC=32 TAG=b81920_n64"]
    if orn_ok:
        head += ["orn text2 hle SAMPLE_N=64 MAX_TOKENS=81920 CONC=32 TAG=b81920",
                 "orn lmeval r1_gpqa_diamond SEED=42 SAMPLE_N=99 TOTAL=198 MAX_TOKENS=81920 CONC=32 TAG=s42_b81920"]
    f.seek(0); f.truncate(); f.write("\n".join(head + out))

"""mm_burst_test.py <model_id> [n=128] [conc=32]: the burst that killed the server on 2026-10-30 21:04 (128 sampled HLE
questions incl. images, 32 at once): every request must come back 200 and the server must stay up. Short answers."""
import sys, json, random, time, concurrent.futures as cf, urllib.request
sys.path.insert(0, "/home/raahem/evals/text2")
from benches import hle
model = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 128; conc = int(sys.argv[3]) if len(sys.argv) > 3 else 32
items = hle.load_items()
sel = [items[i] for i in sorted(random.Random(1234).sample(range(len(items)), n))]
print(f"{n} requests, {sum(1 for it in sel if it['image'])} with images, {conc} at once", flush=True)
def one(it):
    body = {"model": model, "messages": hle.messages(it), "max_tokens": 64, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    req = urllib.request.Request("http://localhost:20000/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status
    except Exception as e:
        return repr(e)[:120]
t0 = time.time()
with cf.ThreadPoolExecutor(conc) as ex:
    res = list(ex.map(one, sel))
ok = sum(1 for r in res if r == 200)
print(f"MM_BURST ok={ok}/{n} {time.time()-t0:.0f}s errors={sorted(set(r for r in res if r != 200))[:3]}")
sys.exit(0 if ok == n else 1)

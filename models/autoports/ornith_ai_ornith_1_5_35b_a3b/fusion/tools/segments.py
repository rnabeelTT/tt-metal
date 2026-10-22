import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
wins, cur = [], None
for x in rows:
    if x["OP TYPE"] == "signpost":
        if x["OP CODE"].endswith("_END"):
            wins.append(cur)
            cur = None
        else:
            cur = []
    elif cur is not None and x["DEVICE ID"] == "0":
        cur.append(x)
SEG = {
    "gdn": [("boundary", [0, 1, 2, 49, 50, 51, 52, 53, 54, 83, 84]), ("mixer", range(3, 49)), ("moe", range(55, 83))],
    "attn": [("boundary", [0, 1, 2, 44, 45, 46, 47, 48, 49, 78, 79]), ("mixer", range(3, 44)), ("moe", range(50, 78))],
}
for name, w in zip(["gdn", "attn"], wins):
    n = len(w) // 5
    print("==", name)
    for seg, idx in SEG[name]:
        tk = to = 0
        c = 0
        for it in range(1, 5):
            step = w[it * n : (it + 1) * n]
            for i in idx:
                x = step[i]
                tk += float(x["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3
                g = float(x["OP TO OP LATENCY [ns]"] or 0) / 1e3
                to += min(g, 20) if i == 0 else g
            c = len(list(idx))
        print(f"  {seg:9s} ops {c:3d} kernel {tk/4:6.1f} us gaps {to/4:6.1f} us total {(tk+to)/4:6.1f} us")

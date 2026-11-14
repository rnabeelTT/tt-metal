import torch, ttnn
dev = ttnn.open_device(device_id=0)
try:
    p = torch.tensor([0, 5, 575, 576, 577, 131071, 1048575, 999999], dtype=torch.int32)
    t = ttnn.from_torch(p, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    P = 576
    want = p % P
    for name, fn in [
        ("remainder_int", lambda: ttnn.remainder(t, P)),
        ("tile_remainder_int", lambda: ttnn.to_layout(ttnn.remainder(ttnn.to_layout(t, ttnn.TILE_LAYOUT), P), ttnn.ROW_MAJOR_LAYOUT)),
        ("fp32_floor", lambda: ttnn.to_layout(ttnn.typecast(ttnn.subtract(ttnn.typecast(ttnn.to_layout(t, ttnn.TILE_LAYOUT), ttnn.float32),
                                                ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.typecast(ttnn.to_layout(t, ttnn.TILE_LAYOUT), ttnn.float32), 1.0 / P)), float(P))), ttnn.int32), ttnn.ROW_MAJOR_LAYOUT)),
    ]:
        try:
            r = ttnn.to_torch(fn()).reshape(-1)[: len(p)].to(torch.int64)
            print(name, "OK" if torch.equal(r, want.to(torch.int64)) else f"WRONG {r.tolist()} want {want.tolist()}")
        except Exception as e:
            print(name, "ERR", str(e).splitlines()[0][:150])
finally:
    ttnn.close_device(dev)

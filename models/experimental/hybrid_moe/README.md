# hybrid_moe

Serve MoE models larger than the device's DRAM on one QB2: each layer's most-used experts on the device, the rest on
the host CPU (int4) or streamed over PCIe, DFlash speculative decoding, elastic context up to 128K.

- **Bringing up a new model: [ADD_A_MODEL.md](ADD_A_MODEL.md)** (with `templates/`)
- Published example: [raahemnabeel/minimax-m2.7-hybrid-qb2](https://huggingface.co/raahemnabeel/minimax-m2.7-hybrid-qb2)
  (`tt-model serve raahemnabeel/minimax-m2.7-hybrid-qb2`)
- Measurements, decisions and open issues: [HANDOFF.md](HANDOFF.md); original plan: [docs/PLAN.md](docs/PLAN.md)

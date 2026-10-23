"""Stage 11: re-aggregate the TTI release report from its saved report data (no device, no server) with the model
spec's known_issues waivers, using TTI's own acceptance and report code (workflow_module.execution path: the
acceptance_criteria_check + build_acceptance_export + ReportGenerator). TTI v0.22.0 has no standalone `reports`
workflow, so this is the equivalent of rerunning only the aggregation over valid raw outputs.
TTI's markdown keeps the raw check of a waived row (FAIL) and lists no waiver reasons, so the markdown is post-edited:
a waived eval row shows "WAIVED (known issue)" in its Accuracy Check cell, and a "Waived rows" list with each reason
follows the acceptance criteria. The data JSON is TTI's own (the reasons are in acceptance_criteria_metadata).
Run from the TTI checkout with its workflow venv: MODEL_SPECS_ENV=dev python 11-tti-regen-report.py REPORT_DATA_JSON OUT_DIR"""
import json
import sys
from pathlib import Path

from report_module.acceptance_criteria import acceptance_criteria_check, build_acceptance_export
from report_module.generator import ReportGenerator
from report_module.schema import ReportSchema
from workflows.model_spec import MODEL_SPECS

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
spec = [v for k, v in MODEL_SPECS.items() if "GLM-5.3-Flash" in k][0]
known = getattr(spec.device_model_spec, "known_issues", None) or getattr(spec, "known_issues", None)
schema = ReportSchema.from_dict({"metadata": d["metadata"], "sections": d["sections"]})
schema.metadata["report_id"] = str(schema.metadata.get("report_id")) + "_waiver-regen"
accepted, blockers, cats = acceptance_criteria_check(
    schema, known_issues=known, model_status=str(spec.status.name if hasattr(spec.status, "name") else spec.status)
)
schema.metadata.update(
    build_acceptance_export(
        accepted, blockers, cats, str(spec.status.name if hasattr(spec.status, "name") else spec.status)
    )
)
res = ReportGenerator().generate(schema, out)
waived = {
    k: v
    for c in (schema.metadata.get("acceptance_criteria_metadata") or {}).get("categories", [])
    for k, v in (c.get("waived") or {}).items()
}
if waived:
    md_path = Path(res.markdown_path)
    lines = md_path.read_text().splitlines()
    tasks = {k.split("\u2014", 1)[-1].strip(): v for k, v in waived.items() if k.startswith("evals:")}
    for i, line in enumerate(lines):
        cells = line.split("|")
        if len(cells) > 2 and cells[1].strip() in tasks and "FAIL" in line:
            lines[i] = line.replace("\u274c FAIL", "WAIVED (known issue)", 1)
    j = next(
        i
        for i, l in enumerate(lines)
        if l.startswith("- All acceptance criteria") or l.startswith("- Acceptance status")
    )
    while j + 1 < len(lines) and lines[j + 1].startswith("- "):
        j += 1
    lines[j + 1 : j + 1] = ["", "Waived rows (TTI `known_issues` of the model spec):", ""] + [
        f"- `{k}`: {v}" for k, v in waived.items()
    ]
    md_path.write_text("\n".join(lines) + "\n")
print("Acceptance:", "PASS" if accepted else "FAIL", "blockers", blockers)
print("report", res.markdown_path)
sys.exit(0 if accepted else 1)

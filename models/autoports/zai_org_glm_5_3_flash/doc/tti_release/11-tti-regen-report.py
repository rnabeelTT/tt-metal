"""Stage 11: re-aggregate the TTI release report from its saved report data (no device, no server) with the model
spec's known_issues waivers, using TTI's own acceptance and report code (workflow_module.execution path: the
acceptance_criteria_check + build_acceptance_export + ReportGenerator). TTI v0.22.0 has no standalone `reports`
workflow, so this is the equivalent of rerunning only the aggregation over valid raw outputs.
Run from the TTI checkout with its workflow venv: MODEL_SPECS_ENV=dev python 11-tti-regen-report.py REPORT_DATA_JSON OUT_DIR"""
import json
import sys

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
print("Acceptance:", "PASS" if accepted else "FAIL", "blockers", blockers)
print("report", res.markdown_path)
sys.exit(0 if accepted else 1)

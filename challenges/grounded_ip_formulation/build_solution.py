"""Dev-only: assemble solution.py = solution_head.py + grammar tool (formulation.py, from MAX_VARS on) + solution_pipeline.py."""
from pathlib import Path
d = Path(__file__).parent
form = (d / "formulation.py").read_text().split("MAX_VARS = 60", 1)[1]
out = (d / "solution_head.py").read_text().rstrip("\n") + "\n\nMAX_VARS = 60" + form.rstrip("\n") + "\n" + (d / "solution_pipeline.py").read_text()
(d / "solution.py").write_text(out)
print(len(out.encode()), "bytes")

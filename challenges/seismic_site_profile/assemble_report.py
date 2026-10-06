"""Stitch final_approach.md from its reviewed sections in the right order."""
from pathlib import Path
order = [f"final_approach_part{i}.md" for i in range(1, 6)]
secs = {}
for f in order:
    if not Path(f).exists():
        continue
    txt = Path(f).read_text()
    # split on "## n." headings, keep the numeric key for ordering
    cur, buf = None, []
    for line in txt.splitlines():
        if line.startswith("## ") and line[3].isdigit():
            if cur is not None:
                secs[cur] = "\n".join(buf).rstrip()
            cur = int(line[3:line.index(".")])
            buf = [line]
        elif line.startswith("# ") and cur is None:
            secs[0] = line
        else:
            buf.append(line)
    if cur is not None:
        secs[cur] = "\n".join(buf).rstrip()
out = [secs.pop(0, "# Final approach - Seismic Site Profile")]
for k in sorted(secs):
    out.append(secs[k])
Path("final_approach.md").write_text("\n\n".join(out) + "\n")
print("final_approach.md:", len(Path('final_approach.md').read_text().splitlines()), "lines,",
      "sections", sorted(secs))

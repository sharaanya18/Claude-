"""Assemble the single self-contained solution.py from the reviewed parts.
Run by hand during development; the shipped file is plain readable source."""
import re, sys
from pathlib import Path

groups = sys.argv[1] if len(sys.argv) > 1 else "ABCDEFGHIJKM"
minutes = sys.argv[2] if len(sys.argv) > 2 else "?"
model_names = sys.argv[3] if len(sys.argv) > 3 else "ordgb,hgb,ord,et,lr,pls,ldak"

head = Path("sol_header_template.py").read_text()
feat = Path("features.py").read_text()
main = Path("sol_main_template.py").read_text()
drv = Path("sol_driver_template.py").read_text()

# strip the feature module's own docstring and numpy import (header has them)
feat = feat[feat.index('"""', feat.index('"""') + 3) + 3:]
feat = feat.replace("import numpy as np\n", "", 1)
# drop the feature module's own group constant: USE_GROUPS in the header is the
# single source of truth for which families the shipped model actually uses
feat = re.sub(r'\nGROUPS = "[A-Z]+"\n', "\n", feat)
feat_doc = '''
# ===========================================================================
# Feature extraction. Physics the representation rests on: a soft surface layer
# over firmer rock resonates near f0 ~ Vs/(4H), amplifies the horizontals far
# more than the vertical there, and attenuates high frequencies. A single
# record is source x path x site; the three components share the source and the
# path, so component RATIOS (H/V) and a spectrum's shape RELATIVE TO ITS OWN
# smooth trend keep the site term while cancelling most of the rest. Amplitude
# is normalised per record, so only shape and inter-component ratios survive.
# Every quantity below is computed from one record in isolation.
# ===========================================================================
'''

head = head.replace('USE_GROUPS = "PLACEHOLDER_GROUPS"', f'USE_GROUPS = "{groups}"')
head = head.replace("N_MINUTES_PLACEHOLDER", minutes)
head = head.replace("PLACEHOLDER_MODELS",
                    "(" + ", ".join(f'"{m}"' for m in model_names.split(",")) + ",)")

out = head + feat_doc + feat + "\n" + main + "\n\nN_FEAT = sum(len(names()[g]) for g in USE_GROUPS)\n\n" + drv
out = re.sub(r"\n{4,}", "\n\n\n", out)
Path("solution.py").write_text(out)
n = len(out.encode())
print(f"solution.py written: {n} bytes, {out.count(chr(10))} lines "
      f"(limit 512000 bytes: {'OK' if n < 512000 else 'TOO BIG'})")

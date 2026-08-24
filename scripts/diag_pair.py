import re
from collections import Counter

log = open("artifacts/v3/cts/naive_s0p_32.log").read()
final1 = len(re.findall(r"final_ok.: 1.0", log))
final0 = len(re.findall(r"final_ok.: 0.0", log))
print("final_ok=1:", final1, "final_ok=0:", final0)
util_vals = re.findall(r"util.: (0\.[0-9]+)", log)
c = Counter(round(float(u), 2) for u in util_vals)
print("util dist:", dict(sorted(c.items())[:12]))

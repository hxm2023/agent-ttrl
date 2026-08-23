import re, glob
contexts = []
for f in sorted(glob.glob('sections/*.tex')) + ['main.tex']:
    lines = open(f, encoding='utf-8').read().splitlines()
    for i, line in enumerate(lines):
        for m in re.finditer(r'\\cite\{([^}]+)\}', line):
            keys = [k.strip() for k in m.group(1).split(',')]
            ctx = ' '.join(lines[max(0, i - 1):i + 2])
            for k in keys:
                contexts.append((k, f, i + 1, ctx[:400]))
with open('.aris/citation-audit/contexts.txt', 'w', encoding='utf-8') as fh:
    for k, fname, line, ctx in contexts:
        fh.write(f"[{k}] {fname}:{line}\n  {ctx}\n")
cited = sorted(set(k for k, _, _, _ in contexts))
print(f"cited keys ({len(cited)}): {cited}")
bib = open('references.bib', encoding='utf-8').read()
bib_keys = re.findall(r'@\w+\{([^,]+),', bib)
uncited = sorted(set(bib_keys) - set(cited))
print(f"uncited bib keys: {uncited}")

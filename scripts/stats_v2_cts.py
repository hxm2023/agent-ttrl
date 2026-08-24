"""v2 CTS statistics: exact two-sided sign-flip on paired per-seed AUPC."""
import json
import random
import sys


def load_aupc(log_path):
    with open(log_path) as f:
        for line in f:
            if "AUPC=" in line:
                return float(line.split("AUPC=")[-1].split()[0])
    return None


def main():
    base = "artifacts/v2/cts" if len(sys.argv) < 2 else sys.argv[1]
    n_tasks = 16
    f, n, e = [], [], []
    for s in range(8):
        fa = load_aupc(f"{base}/frozen_s{s}_{n_tasks}.log")
        na = load_aupc(f"{base}/naive_s{s}_{n_tasks}.log")
        try:
            ea = load_aupc(f"{base}/egc_s{s}_{n_tasks}.log")
        except FileNotFoundError:
            ea = None
        if fa is not None and na is not None:
            f.append(fa); n.append(na)
        if fa is not None and ea is not None:
            e.append(ea)
    df = [n[i] - f[i] for i in range(len(f))]
    de = [e[i] - f[i] for i in range(len(e))]
    print(f"{n_tasks}-task: frozen mean {sum(f)/len(f):.4f} | naive mean {sum(n)/len(n):.4f} | egc mean {sum(e)/len(e):.4f}" if e else
          f"{n_tasks}-task: frozen mean {sum(f)/len(f):.4f} | naive mean {sum(n)/len(n):.4f}")
    print(f"naive-frozen deltas: {[round(x, 4) for x in df]} mean {sum(df)/len(df):+.4f} positive {sum(1 for x in df if x > 0)}/{len(df)}")
    if len(df) >= 4:
        obs = sum(df) / len(df)
        rng = random.Random(0)
        hits = 0
        for _ in range(200000):
            perm = 0.0
            for d in df:
                perm += d if rng.random() < 0.5 else -d
            if abs(perm / len(df)) >= abs(obs):
                hits += 1
        print(f"exact two-sided sign-flip p = {hits/200000:.4f}")
    if e:
        print(f"egc-frozen deltas: {[round(x, 4) for x in de]} mean {sum(de)/len(de):+.4f} positive {sum(1 for x in de if x > 0)}/{len(de)}")
        if len(de) >= 4:
            obs = sum(de) / len(de)
            rng = random.Random(1)
            hits = 0
            for _ in range(200000):
                perm = 0.0
                for d in de:
                    perm += d if rng.random() < 0.5 else -d
                if abs(perm / len(de)) >= abs(obs):
                    hits += 1
            print(f"egc exact two-sided sign-flip p = {hits/200000:.4f}")


if __name__ == "__main__":
    main()

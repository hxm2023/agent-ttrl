import json, os, sys
kb_dir = os.path.dirname(os.path.abspath(__file__))
papers_dir = os.path.join(kb_dir, "papers")

def search(query, top_k=10):
    results = []
    query_terms = query.lower().split()
    for fname in ["domain_overview.md", "metrics_and_baselines.md", "field_conventions.md"]:
        fpath = os.path.join(kb_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as f:
                text = f.read().lower()
            score = sum(text.count(t) for t in query_terms)
            if score > 0:
                results.append({"source": fname, "score": score, "type": "domain"})
    for fname in os.listdir(papers_dir):
        if not fname.endswith('.json'): continue
        with open(os.path.join(papers_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        text = data["full_text"].lower()
        score = sum(text.count(t) for t in query_terms)
        if score > 0:
            idx = text.find(query_terms[0]) if query_terms else 0
            snippet = data["full_text"][max(0,idx-150):idx+150]
            results.append({"source": data["slug"], "score": score, "snippet": snippet, "type": "paper"})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    for r in search(" ".join(sys.argv[1:])):
        print(f"\n[{r['type']}] {r['source']} (score={r['score']})")
        if r['type'] == 'paper':
            print(f"  ...{r['snippet'][:300]}...")

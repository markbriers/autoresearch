# Academic Paper Search (OpenAlex)

Search 200M+ academic papers via the OpenAlex API. Relevance-ranked, fast (10 req/sec free), no API key required.

The user provides a query, paper ID/DOI, or describes what they're looking for. Execute the appropriate search below.

**Base URL:** `https://api.openalex.org`

## Operations

### 1. Search papers by relevance

Use Bash with curl (faster than WebFetch, avoids redirect issues):

```bash
curl -s "https://api.openalex.org/works?search=YOUR+QUERY+HERE&filter=publication_year:2023-2026&per_page=10&sort=relevance_score:desc&select=id,title,publication_year,cited_by_count,doi,abstract_inverted_index,primary_topic,authorships" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for p in d.get('results',[]):
    aii = p.get('abstract_inverted_index') or {}
    if aii:
        words = [''] * (max(max(v) for v in aii.values()) + 1)
        for word, positions in aii.items():
            for pos in positions:
                words[pos] = word
        abstract = ' '.join(w for w in words if w)[:300]
    else:
        abstract = '(no abstract)'
    authors = ', '.join(a['author']['display_name'] for a in (p.get('authorships') or [])[:3])
    topic = (p.get('primary_topic') or {}).get('display_name', '?')
    doi = p.get('doi','') or ''
    arxiv_id = doi.replace('https://doi.org/10.48550/arxiv.','') if 'arxiv' in doi.lower() else ''
    print(f\"## {p['title']}\")
    print(f\"Year: {p['publication_year']} | Cites: {p['cited_by_count']} | Topic: {topic}\")
    print(f\"Authors: {authors}\")
    if arxiv_id: print(f\"arXiv: {arxiv_id}\")
    print(f\"DOI: {doi}\")
    print(f\"Abstract: {abstract}...\")
    print()
"
```

**Useful filters** (append to `filter=` with commas):
- `publication_year:2024-2026` — year range
- `cited_by_count:>10` — minimum citations
- `topics.subfield.display_name:Artificial Intelligence` — by subfield
- `type:article` — only journal articles
- `open_access.is_oa:true` — open access only

### 2. Get paper details by DOI or OpenAlex ID

```bash
curl -s "https://api.openalex.org/works/doi:10.48550/arXiv.2406.06811?select=id,title,publication_year,cited_by_count,doi,abstract_inverted_index,authorships,referenced_works,primary_topic" | python3 -m json.tool | head -80
```

Supported ID formats:
- DOI: `doi:10.1234/example`
- arXiv DOI: `doi:10.48550/arXiv.2406.06811`
- OpenAlex ID: `W4399597690`

### 3. Find papers that cite a given paper

First get the OpenAlex ID, then query citations:

```bash
# Step 1: Get OpenAlex ID from DOI
OA_ID=$(curl -s "https://api.openalex.org/works/doi:10.48550/arXiv.XXXX.XXXXX?select=id" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'].split('/')[-1])")

# Step 2: Get citing papers, sorted by citation count
curl -s "https://api.openalex.org/works?filter=cites:$OA_ID&per_page=10&sort=cited_by_count:desc&select=id,title,publication_year,cited_by_count,doi" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Total citing papers: {d[\"meta\"][\"count\"]}')
for p in d['results']:
    print(f\"  {p['publication_year']} [{p['cited_by_count']} cites] {p['title']}\")
"
```

### 4. Get a paper's references (what it cites)

```bash
curl -s "https://api.openalex.org/works/doi:10.48550/arXiv.XXXX.XXXXX?select=id,title,referenced_works" | python3 -c "
import sys, json
d = json.load(sys.stdin)
refs = d.get('referenced_works',[])
print(f'{d[\"title\"]} cites {len(refs)} papers')
# Fetch details of first 10 references
if refs:
    ids = '|'.join(r.split('/')[-1] for r in refs[:10])
    import subprocess
    out = subprocess.run(['curl','-s',f'https://api.openalex.org/works?filter=openalex:{ids}&select=id,title,publication_year,cited_by_count,doi'], capture_output=True, text=True)
    d2 = json.loads(out.stdout)
    for p in d2.get('results',[]):
        print(f\"  {p['publication_year']} [{p['cited_by_count']} cites] {p['title']}\")
"
```

### 5. Cross-domain discovery — search without field filters

To find analogies from outside ML (signal processing, control theory, neuroscience, etc.), omit the topic filter and use conceptual queries:

```bash
curl -s "https://api.openalex.org/works?search=spectral+normalization+gradient+flow+stability&filter=publication_year:2020-2026,cited_by_count:>5&per_page=10&select=id,title,publication_year,cited_by_count,doi,primary_topic" | python3 -c "
import sys, json
for p in json.load(sys.stdin).get('results',[]):
    topic = (p.get('primary_topic') or {}).get('display_name', '?')
    print(f\"{p['publication_year']} [{p['cited_by_count']} cites] [{topic}] {p['title']}\")
"
```

## Bridging to Full Paper Text

OpenAlex provides metadata and abstracts but not full paper text. To read the full methods section:

1. Get the arxiv ID from the DOI field (strip `https://doi.org/10.48550/arXiv.`)
2. Use the arxiv MCP server: `download_paper` then `read_paper` with that ID

This gives you: **OpenAlex for fast, relevant discovery → arxiv MCP for deep reading**.

## Rate Limits

- Without API key: ~$0.01/day credit (enough for ~100 search queries)
- With free API key (get at openalex.org/settings/api): $1/day (~10,000 searches)
- Throughput: up to 10 requests/second
- No shared pool — your rate limit is yours

## Tips

- **Abstracts are inverted indices** — use the Python reconstruction snippet above
- **`related_works` field is unreliable** — use citation graph traversal instead (cites/cited_by)
- **Search is relevance-ranked by default** — much better than arxiv keyword matching
- **Chain: search → find good paper → get its references → search within those** for deep literature exploration
- **For the autoresearch loop**: search OpenAlex first, find arxiv IDs, then use arxiv MCP to read full papers

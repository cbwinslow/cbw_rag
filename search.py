#!/usr/bin/env python3
"""
CBW RAG Search - Vector, text, and hybrid search over indexed files.
"""

import os, sys, json, argparse
from typing import List, Optional

import psycopg2
import psycopg2.extras
import ollama

DB_DSN = os.environ.get("CBW_RAG_DATABASE", "postgresql://cbwinslow:123qweasd@localhost:5432/cbw_rag")
EMBEDDING_MODEL = os.environ.get("CBW_RAG_EMBEDDING_MODEL", "nomic-embed-text")

def get_conn():
    return psycopg2.connect(DB_DSN)

def embed_query(query, model=EMBEDDING_MODEL):
    r = ollama.embeddings(model=model, prompt=query)
    return r['embedding']

def search_vector(query, limit=10, category=None, ext=None, model=EMBEDDING_MODEL):
    emb = embed_query(query, model)
    emb_str = '[' + ','.join(str(x) for x in emb) + ']'
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    where = ["fc.embedding IS NOT NULL"]
    cat_params = []
    if category:
        where.append("f.metadata->>'file_category'=%s"); cat_params.append(category)
    if ext:
        where.append("f.file_extension=%s"); cat_params.append(ext if ext.startswith('.') else f'.{ext}')
    w = ' AND '.join(where)
    cur.execute(f"""
        SELECT fc.id AS chunk_id, f.file_path AS source_path, f.file_name, f.file_extension,
            f.metadata->>'file_category' AS file_category, f.metadata->>'detected_language' AS detected_language,
            fc.chunk_text AS content, fc.chunk_index,
            fc.line_start AS start_line, fc.line_end AS end_line,
            1 - (fc.embedding <=> %s::vector) AS similarity
        FROM file_chunks fc
        JOIN files f ON f.id=fc.file_id
        WHERE {w}
        ORDER BY fc.embedding <=> %s::vector LIMIT %s
    """, cat_params + [emb_str, emb_str, limit])
    results = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return results

def search_text(query, limit=10, category=None):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    where = []
    cat_params = []
    if category:
        where.append("f.metadata->>'file_category'=%s"); cat_params.append(category)
    w = ' AND '.join(where) if where else '1=1'
    cur.execute(f"""
        SELECT fc.id AS chunk_id, f.file_path AS source_path, f.file_name, f.file_extension,
            f.metadata->>'file_category' AS file_category, f.metadata->>'detected_language' AS detected_language,
            fc.chunk_text AS content, fc.chunk_index,
            fc.line_start AS start_line, fc.line_end AS end_line,
            ts_rank(to_tsvector('english',fc.chunk_text), plainto_tsquery('english',%s)) AS rank
        FROM file_chunks fc
        JOIN files f ON f.id=fc.file_id
        WHERE to_tsvector('english',fc.chunk_text) @@ plainto_tsquery('english',%s) AND {w}
        ORDER BY rank DESC LIMIT %s
    """, [query, query] + cat_params + [limit])
    results = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return results

def search_hybrid(query, limit=10, vec_w=0.7, txt_w=0.3, category=None):
    emb = embed_query(query)
    emb_str = '[' + ','.join(str(x) for x in emb) + ']'
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    where = ["fc.embedding IS NOT NULL"]
    cat_params = []
    if category:
        where.append("f.metadata->>'file_category'=%s"); cat_params.append(category)
    w = ' AND '.join(where)
    cur.execute(f"""
        WITH vr AS (
            SELECT fc.id AS chunk_id, 1-(fc.embedding<=>%s::vector) AS vs,
                ROW_NUMBER() OVER (ORDER BY fc.embedding<=>%s::vector) AS vr
            FROM file_chunks fc
            JOIN files f ON f.id=fc.file_id
            WHERE {w}
            ORDER BY fc.embedding<=>%s::vector LIMIT 200
        ), tr AS (
            SELECT fc.id AS chunk_id,
                ts_rank(to_tsvector('english',fc.chunk_text),plainto_tsquery('english',%s)) AS ts,
                ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english',fc.chunk_text),plainto_tsquery('english',%s)) DESC) AS tr
            FROM file_chunks fc
            JOIN files f ON f.id=fc.file_id
            WHERE to_tsvector('english',fc.chunk_text) @@ plainto_tsquery('english',%s)
            ORDER BY ts DESC LIMIT 200
        )
        SELECT COALESCE(v.chunk_id,t.chunk_id) AS chunk_id,
            COALESCE(v.vs,0)*%s+COALESCE(t.ts,0)*%s AS score,
            f.file_path AS source_path, f.file_name, f.file_extension,
            f.metadata->>'file_category' AS file_category,
            fc.chunk_text AS content, fc.chunk_index, fc.line_start AS start_line, fc.line_end AS end_line
        FROM vr v FULL OUTER JOIN tr t ON v.chunk_id=t.chunk_id
        JOIN file_chunks fc ON fc.id=COALESCE(v.chunk_id,t.chunk_id)
        JOIN files f ON f.id=fc.file_id
        ORDER BY score DESC LIMIT %s
    """, [emb_str, emb_str, emb_str] + cat_params + [query, query, query] + [vec_w, txt_w, limit])
    results = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return results

def show_stats():
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            (SELECT COUNT(*) FROM files) AS total_files,
            (SELECT COUNT(*) FROM file_chunks) AS total_chunks,
            (SELECT COUNT(*) FROM file_chunks WHERE embedding IS NOT NULL) AS embeddings_768,
            pg_size_pretty(pg_total_relation_size('files')) AS files_size,
            pg_size_pretty(pg_total_relation_size('file_chunks')) AS chunks_size
    """)
    s = dict(cur.fetchone())
    cur.execute("SELECT metadata->>'file_category' AS file_category, COUNT(*) c FROM files WHERE metadata->>'file_category' IS NOT NULL GROUP BY metadata->>'file_category' ORDER BY c DESC")
    cats = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT metadata->>'detected_language' AS detected_language, COUNT(*) c FROM files WHERE metadata->>'detected_language' IS NOT NULL GROUP BY metadata->>'detected_language' ORDER BY c DESC LIMIT 15")
    langs = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()

    print(f"\n{'='*50}")
    print(f"  CBW RAG Database")
    print(f"{'='*50}")
    print(f"  Files:       {s['total_files']}")
    print(f"  Chunks:      {s['total_chunks']}")
    print(f"  Embeddings:  {s['embeddings_768']} (768d)")
    print(f"  Sizes:       files={s['files_size']} chunks={s['chunks_size']}")
    if cats:
        print(f"\n  By Category:")
        for c in cats: print(f"    {c['file_category'] or '?':15s} {c['c']:>6d}")
    if langs:
        print(f"\n  By Language:")
        for l in langs: print(f"    {l['detected_language']:15s} {l['c']:>6d}")

def format_results(results, verbose=False):
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        score = r.get('similarity') or r.get('score') or r.get('rank', 0)
        lines = f":{r['start_line']}-{r['end_line']}" if r.get('start_line') else ''
        print(f"\n  [{i}] {r['source_path']}{lines}")
        print(f"      Score: {score:.4f} | {r.get('file_category','?')}/{r.get('detected_language','?')}")
        content = r['content']
        if not verbose and len(content) > 300: content = content[:300] + '...'
        for line in content.split('\n')[:8]:
            print(f"      {line}")

def main():
    parser = argparse.ArgumentParser(description='CBW RAG Search')
    parser.add_argument('query', nargs='?')
    parser.add_argument('-n', '--limit', type=int, default=10)
    parser.add_argument('--vector-only', action='store_true')
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--type', help='Filter: code, document, config')
    parser.add_argument('--model', default=EMBEDDING_MODEL)
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.stats:
        show_stats(); return

    if args.text_only:
        fn = lambda q, n: search_text(q, n, category=args.type)
    elif args.vector_only:
        fn = lambda q, n: search_vector(q, n, category=args.type, model=args.model)
    else:
        fn = lambda q, n: search_hybrid(q, n, category=args.type)

    if args.interactive:
        print("CBW RAG Search - Interactive (type 'quit' or 'stats')")
        while True:
            try:
                q = input("rag> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!"); break
            if not q: continue
            if q in ('quit','exit','q'): break
            if q == 'stats': show_stats(); continue
            format_results(fn(q, args.limit), args.verbose)
        return

    if not args.query:
        parser.print_help(); return

    format_results(fn(args.query, args.limit), args.verbose)

if __name__ == '__main__':
    main()

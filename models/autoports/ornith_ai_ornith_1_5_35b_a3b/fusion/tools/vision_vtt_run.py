"""Run one image benchmark against an OpenAI-compatible server and score it.

  python -m vtt.run <bench> --out DIR --base-url URL --model ID --concurrency N [--phase all|infer|score]
  python -m vtt.run <bench> --out DIR --write-stub-key FILE      # gold answers for the stub server
  python -m vtt.run <bench> --out DIR --render-only 1            # print one rendered request

Env: MAX_TOKENS (32768) TEMPERATURE (0.6) TOP_P (0.95) TOP_K (20) PRESENCE_PENALTY (unset)
     LIMIT (first N selected items per sub-dataset, smoke tests) REQUEST_TIMEOUT (14400 s)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault('LMUData', os.path.join(ROOT, 'LMUData'))
os.environ.setdefault('PRED_FORMAT', 'tsv')
os.environ.setdefault('EVAL_FORMAT', 'csv')
# Belt and braces: anything that still tries an API judge hits a dead local port.
for k in ('OPENAI_API_BASE', 'OPENAI_BASE_URL', 'DASHSCOPE_API_BASE', 'SiliconFlow_API_BASE'):
    os.environ[k] = 'http://127.0.0.1:9/v1/chat/completions'
os.environ.pop('OPENAI_API_KEY_JUDGE', None)

from . import common  # noqa: E402
from .common import Cache, build_messages, infer_all, request_fingerprint  # noqa: E402

SAMPLES_DIR = os.path.join(ROOT, 'samples')
EMPTY = '\u2205'


def get_bench(name):
    from .benches import REGISTRY
    from . import extra_benches  # noqa: F401  (registers custom-data benchmarks)
    if name not in REGISTRY:
        sys.exit(f'unknown bench {name}; have: {", ".join(sorted(REGISTRY))}')
    return REGISTRY[name]()


def item_key(sub, idx, r):
    return f'{sub}::{idx}' + (f'::r{r}' if r else '')


def load_items(bench, limit=None):
    """-> list of (key, sub, ds, line, repeat). Sample ids are fixed in samples/<bench>.json."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    sample_file = os.path.join(SAMPLES_DIR, f'{bench.name}.json')
    saved = json.load(open(sample_file)) if os.path.exists(sample_file) else None
    subsets, out, picked = {}, [], []
    for sub, ds in bench.load():
        data = bench.filter(sub, ds.data)
        data = data.assign(index=[str(x) for x in data['index']])
        if saved is not None and sub in saved['ids']:
            ids = saved['ids'][sub]
            missing = set(ids) - set(data['index'])
            assert not missing, f'{len(missing)} saved sample ids missing from {sub}'
        else:
            ids = bench.select(sub, data)
        subsets[sub] = {'ids': ids, 'population': len(data)}
        if limit:
            ids = bench.limit(sub, data, ids, limit) if hasattr(bench, 'limit') else ids[:limit]
        picked.append((sub, ds, data, ids))
    # SAMPLE_N: run-size cap (1-2 h runs): a fixed random subset (seed 1234) of the frozen sample, split over the
    # sub-datasets in proportion to their size
    sample_n = int(os.environ['SAMPLE_N']) if os.environ.get('SAMPLE_N') else None
    total = sum(len(x[3]) for x in picked)
    for sub, ds, data, ids in picked:
        if sample_n and total > sample_n:
            ids = common.sample_ids(ids, max(1, round(sample_n * len(ids) / total)))
            subsets[sub]['sample_n'] = len(ids)
        by = data.set_index('index', drop=False)
        for i in ids:
            line = by.loc[i]
            for r in range(bench.repeats):
                out.append((item_key(sub, i, r), sub, ds, line, r))
    if saved is None:
        json.dump({'bench': bench.name, 'seed': 1234,
                   'rule': 'fixed random sample of 1000 when the set has > ~3000 items (see bench notes for strata/groups)',
                   'ids': {k: v['ids'] for k, v in subsets.items()},
                   'population': {k: v['population'] for k, v in subsets.items()}},
                  open(sample_file, 'w'), indent=0)
    return out, subsets, sample_file


def render(bench, ds, line):
    parts = bench.parts(line, ds)
    system = bench.system_prompt() if hasattr(bench, 'system_prompt') else bench.system
    msgs, infos = build_messages(parts, *bench.pixels, system=system)
    return msgs, bench.extra_body(line, ds), {'images': infos}


def short(msgs):
    """Printable copy of a request (base64 elided)."""
    out = []
    for m in msgs:
        c = m['content']
        if isinstance(c, list):
            c = [({'type': 'image_url', 'image_url': {'url': p['image_url']['url'][:40] + f'...<{len(p["image_url"]["url"])} chars>'}}
                  if p['type'] == 'image_url' else p) for p in c]
        out.append({'role': m['role'], 'content': c})
    return out


def cfg_from_env(a):
    return {
        'base_url': a.base_url.rstrip('/'), 'model': a.model, 'concurrency': a.concurrency,
        'max_tokens': int(os.environ.get('MAX_TOKENS', 32768)),
        'temperature': float(os.environ.get('TEMPERATURE', 0.6)),
        'top_p': float(os.environ.get('TOP_P', 0.95)),
        'top_k': int(os.environ.get('TOP_K', 20)),
        'presence_penalty': float(os.environ['PRESENCE_PENALTY']) if os.environ.get('PRESENCE_PENALTY') else None,
        'timeout': float(os.environ.get('REQUEST_TIMEOUT', 14400)),
        'retries': int(os.environ.get('RETRIES', 5)),
        'keep_reasoning': os.environ.get('KEEP_REASONING', '0') == '1',
    }


def to_jsonable(x):
    import numpy as np
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient='records')
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def score(bench, items, cache: Cache, out_dir):
    import pandas as pd
    from .judge import JudgeStore, install
    store = JudgeStore(os.path.join(out_dir, 'judge'), bench.name, bench.verdict_format)
    install(store)
    work = os.path.join(out_dir, 'score')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    groups = {}
    for key, sub, ds, line, r in items:
        groups.setdefault((sub, r), (ds, []))[1].append((key, line))
    results, ctx, stats = {}, [], {'n': 0, 'errors': 0, 'truncated': 0, 'empty': 0, 'server_downscaled_images': 0}
    for (sub, r), (ds, rows) in groups.items():
        recs = []
        for key, line in rows:
            c = cache.done.get(key, {})
            content = c.get('content') or ''
            stats['n'] += 1
            stats['errors'] += c.get('error') is not None or key not in cache.done
            stats['truncated'] += c.get('finish_reason') == 'length'
            stats['empty'] += not content.strip()
            ims = ((c.get('meta') or {}).get('images') or [])
            for im in ims:
                stats['server_downscaled_images'] += bool(im.get('server_downscale'))
            stats['items_images_tiled'] = stats.get('items_images_tiled', 0) + any(im.get('tiled_from') for im in ims)
            d = {k: v for k, v in line.to_dict().items() if k != 'image'}
            pred = bench.postprocess(content, line)
            # Empty content (failed request, or thinking ran out of tokens before an answer) is scored as
            # the token '∅': an empty cell becomes NaN on the scorers' TSV reload and crashes some of them.
            d['prediction'] = pred if str(pred).strip() else EMPTY
            recs.append(d)
            ctx.append({'id': key, 'question': d.get('question'), 'reference': d.get('answer', d.get('answers')),
                        'model_answer': d['prediction']})
        df = pd.DataFrame(recs)
        name = sub + (f'_r{r}' if r else '')
        sdir = os.path.join(work, name)
        os.makedirs(sdir)
        eval_file = os.path.join(sdir, f'vtt_{sub}.tsv')
        from vlmeval.smp import dump
        dump(df, eval_file)
        jm = bench.judge_model
        if jm is None and hasattr(ds, 'get_default_judge_model'):
            try:
                jm = ds.get_default_judge_model({})
            except Exception:
                jm = getattr(ds, 'DEFAULT_JUDGE_MODEL', None)
        kw = dict(nproc=8, **bench.score_kwargs())
        if jm:
            kw['model'] = jm
        print(f'scoring {name} ({len(df)} rows) judge={jm}', flush=True)
        try:
            res = bench.evaluate(ds, eval_file, kw) if hasattr(bench, 'evaluate') else ds.evaluate(eval_file, **kw)
        except Exception:
            if not store.pending:
                raise
            # official scorer cannot finish while judge verdicts are missing (e.g. 0/0 in an F1): provisional
            traceback.print_exc()
            print(f'scorer incomplete for {name}: judge verdicts pending', flush=True)
            res = None
        results[sub if not r else f'{sub}#r{r}'] = res
    pending_path = store.write_pending(ctx)
    return results, stats, store, pending_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bench')
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-url', default='http://localhost:20000/v1')
    ap.add_argument('--model', default='Qwen/Qwen3.5-122B-A10B')
    ap.add_argument('--concurrency', type=int, default=32)
    ap.add_argument('--phase', default='all', choices=['all', 'infer', 'score'])
    ap.add_argument('--write-stub-key', default=None)
    ap.add_argument('--stub-mode', default='gold', choices=['gold', 'garbage'])
    ap.add_argument('--render-only', type=int, default=0)
    a = ap.parse_args()

    a.out = os.path.abspath(a.out)
    bench = get_bench(a.bench)
    os.makedirs(a.out, exist_ok=True)
    limit = int(os.environ['LIMIT']) if os.environ.get('LIMIT') else None
    prev_cfg = os.path.join(a.out, 'run_config.json')
    if limit is None and a.phase == 'score' and os.path.exists(prev_cfg):
        limit = json.load(open(prev_cfg)).get('limit')   # re-score exactly what was run
    t0 = time.time()
    items, subsets, sample_file = load_items(bench, limit)
    desc = ', '.join('%s: %d/%d' % (k, len(v['ids']), v['population']) for k, v in subsets.items())
    print(f'{bench.name}: {len(items)} requests ({desc}) repeats={bench.repeats} sample_file={sample_file} '
          f'load={time.time() - t0:.0f}s', flush=True)
    shutil.copy(sample_file, os.path.join(a.out, 'sample_ids.json'))

    if a.render_only:
        for key, sub, ds, line, r in items[:a.render_only]:
            msgs, extra, meta = render(bench, ds, line)
            print('RENDERED', key, json.dumps({'messages': short(msgs), 'extra_body': extra, 'meta': meta}, ensure_ascii=False, indent=1)[:6000])
            print('GOLD', bench.gold(line, ds))
        return

    if a.write_stub_key:
        rng = random.Random(0)
        n = 0
        with open(a.write_stub_key, 'w') as f:
            for key, sub, ds, line, r in items:
                msgs, extra, meta = render(bench, ds, line)
                g = bench.gold(line, ds)
                bad = bench.garbage(line, rng)
                f.write(json.dumps({'fp': request_fingerprint(msgs), 'id': key, 'gold': g, 'garbage': bad}, ensure_ascii=False) + '\n')
                n += 1
        print(f'wrote {n} stub keys to {a.write_stub_key}')
        return

    cfg = cfg_from_env(a)
    cache = Cache(os.path.join(a.out, 'responses.jsonl'))
    meta = {'bench': bench.name, 'card': bench.card, 'cfg': {k: v for k, v in cfg.items()}, 'notes': bench.notes,
            'subsets': {k: {'n': len(v['ids']), 'population': v['population']} for k, v in subsets.items()},
            'limit': limit, 'repeats': bench.repeats, 'pixels': bench.pixels}
    if a.phase != 'score' or not os.path.exists(prev_cfg):
        json.dump(meta, open(prev_cfg, 'w'), indent=1)

    if a.phase in ('all', 'infer'):
        jobs = [(key, (lambda b=bench, d=ds, l=line: render(b, d, l))) for key, sub, ds, line, r in items]
        t1 = time.time()
        infer_all(jobs, cfg, cache)
        print(f'inference wall {time.time() - t1:.0f}s', flush=True)
        bad = [k for k, *_ in items if not cache.ok(k)]
        if bad:
            print(f'WARNING {len(bad)} requests failed (e.g. {bad[0]}: {cache.done[bad[0]].get("error")}); '
                  f'rerun the same command to retry them', flush=True)

    if a.phase in ('all', 'score'):
        try:
            results, stats, store, pending = score(bench, items, cache, a.out)
        except Exception:
            traceback.print_exc()
            print(f'RESULT {bench.name} status=score_error n={len(items)}')
            sys.exit(1)
        bench.n_items = stats['n'] // max(1, bench.repeats)
        try:
            metric, value, extra = bench.headline(results)
        except Exception:
            if not store.pending:
                raise
            metric, value, extra = 'headline', None, {'note': 'not computable until judge verdicts are ingested'}
        status = 'provisional' if pending else 'final'
        summary = {'bench': bench.name, 'metric': metric, 'value': value, 'card': bench.card, 'status': status,
                   'stats': stats, 'judge': {'calls': store.calls, 'answered_from_judgments': store.hits,
                                             'pending_prompts': len(store.pending), 'pending_file': pending},
                   'extra': to_jsonable(extra), 'raw': to_jsonable(results)}
        json.dump(summary, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, default=str)
        if pending:
            print(f'JUDGE_PENDING {pending}')
        v = 'NA' if value is None else f'{value:.2f}'
        print(f'RESULT {bench.name} {metric}={v} n={stats["n"]} errors={stats["errors"]} truncated={stats["truncated"]} '
              f'empty={stats["empty"]} judge_pending={len(store.pending)} status={status} card={bench.card.replace(" ", "_")}')


if __name__ == '__main__':
    main()

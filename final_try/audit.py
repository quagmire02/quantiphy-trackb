"""CPU-only audit of the supplied artifacts; never used to choose test predictions."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.parse import finalize_question
from pipeline.official import parse_number
from pipeline.scoring import mra


def main():
    df = pd.read_parquet(ROOT / 'data/test/test_dataset.parquet').reset_index(drop=True)
    df['id'] = df.index
    df['category'] = df.inference_type.str[0] + df.video_type.str[1]
    inputs = Path(__file__).parent / 'inputs'
    a = pd.read_csv(inputs / '4b_measure.csv').set_index('id').reindex(df.id)
    b = pd.read_csv(inputs / '32b_measure.csv').set_index('id').reindex(df.id)
    p = {r['id']: r for r in map(json.loads, (inputs / '32b_parse_question.jsonl').read_text().splitlines())}
    raw = {r['id']: r for r in map(json.loads, (inputs / '32b_direct.jsonl').read_text().splitlines())}
    lines = []
    def emit(x):
        lines.append('\n'.join(line.rstrip() for line in str(x).splitlines()))
        print(x)
    emit('Coverage (valid = finite and positive; test IDs are zero based):')
    av, bv = np.isfinite(a.value) & (a.value > 0), np.isfinite(b.value) & (b.value > 0)
    emit(pd.crosstab(np.where(av, '4B measured', '4B failed') + np.where(bv, ' -> 32B measured', ' -> 32B failed'), df.category).to_string())
    emit('\n32B failure reasons by category:')
    reason = b.reason.fillna('').str.replace(r'\[.*\]|-?\d+\.\d+', '', regex=True)
    emit(pd.crosstab(reason[~bv], df.category[~bv]).to_string())
    invalid = [i for i, r in p.items() if not r.get('question_ok')]
    emit(f'\nQuestion cache: {len(p)} records; invalid {len(invalid)}')
    emit(pd.Series([p[i].get('target_quantity') for i in invalid]).value_counts(dropna=False).to_string())
    emit('Invalid question examples:')
    for i in invalid[:20]:
        emit((i, df.loc[i, 'question'], p[i]))
    badraw = [r for r in raw.values() if not np.isfinite(parse_number(r.get('reply'))) or parse_number(r.get('reply')) <= 0]
    emit(f'\nDirect cache: {len(raw)} records; unusable replies {len(badraw)}; 32B failures with NO raw record {sum(i not in raw for i in df.id[~bv])}')
    emit('Unusable raw replies (first 15):')
    for r in badraw[:15]:
        emit((r['id'], repr(r.get('reply'))))
    emit('All raw reply values: ' + str(pd.Series([r.get('value') for r in raw.values()]).describe().to_dict()))
    both = av & bv
    emit('\nMeasurements that differ by >2x:')
    emit(pd.crosstab(df.category[both], (abs(np.log(b.value[both] / a.value[both])) > np.log(2))).to_string())
    val = pd.read_csv(ROOT / 'data/validation/validation_dataset.csv')
    norm = lambda q: re.sub(r'\s+', ' ', str(q)).strip().lower()
    keys = dict(zip(zip(val.video_id.str.strip(), val.question.map(norm)), val.ground_truth_posterior))
    truth = np.array([keys.get((v.strip(), norm(q)), np.nan) for v, q in zip(df.video_id, df.question)])
    has = np.isfinite(truth)
    emit(f'\nDevelopment-only labelled overlap: {has.sum()} items / {df.loc[has, "video_id"].nunique()} videos; correlated and previously used for development.')
    v16 = pd.read_csv(inputs / 'v16_sources.csv').parsed_value.values
    v17 = pd.read_csv(inputs / 'v17_sources.csv').parsed_value.values
    for name, pred in [('v16', v16), ('v17', v17), ('4B measurement', a.value.values), ('32B measurement', b.value.values)]:
        score = mra(pred[has], truth[has])
        cats = df.category.values[has]
        by = {c: round(float(score[cats == c].mean() * 100), 2) for c in ['S2', 'D2', 'S3', 'D3']}
        emit(f'{name}: {by}, macro={np.mean(list(by.values())):.2f}')
    emit('Paired overlap where both measured:')
    for c in ['S2', 'D2', 'S3', 'D3']:
        mask = has & both.values & (df.category.values == c)
        emit(f'{c} n={mask.sum()}: 4B={mra(a.value.values[mask], truth[mask]).mean()*100:.2f}, 32B={mra(b.value.values[mask], truth[mask]).mean()*100:.2f}')
    out = Path(__file__).parent / 'audit_results.txt'
    out.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

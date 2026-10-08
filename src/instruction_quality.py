"""Prepare evidence-backed candidates, audit 15 random pairs, publish reviewed SFT data.

Automatic text checks are screening heuristics; humans must assess meaning and grounding.
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from src import config

REVISION = config.DATA_DIR / 'instruction_revision'
TYPES = {'factual', 'procedural', 'comparative'}
REVIEW_FIELDS = ['grounded', 'own_words', 'concise', 'correct_question_type']


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.lower()))


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_pair(row, corpus):
    errors=[]
    answer=row.get('response', ''); words=answer.split()
    if not 30 <= len(words) <= 90: errors.append('response_not_30_to_90_words')
    # Periods in decimal numbers do not split; abbreviations may need manual review.
    sentences=[x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z])', answer.strip()) if x]
    if not 1 <= len(sentences) <= 3: errors.append('response_not_1_to_3_sentences')
    if not row.get('instruction', '').strip().endswith('?'): errors.append('not_a_question')
    if row.get('question_type') not in TYPES: errors.append('missing_question_type')
    if not row.get('template_family', '').strip(): errors.append('missing_template_family')
    if not row.get('group'): errors.append('missing_evidence_group')
    source=corpus.get(row.get('source'))
    evidence=row.get('evidence_quote', '')
    if source is None: errors.append('unknown_source')
    elif len(evidence.split()) < 8 or normalized(evidence) not in normalized(source):
        errors.append('evidence_not_found_in_source')
    if source and normalized(answer) in normalized(source): errors.append('verbatim_response')
    if re.search(r'\.{5,}|_{5,}',answer): errors.append('contents_or_form_noise')
    return errors


def prepare(corpus_dir=config.DOMAIN_CORPUS_DIR):
    """Export evidence requests; no paid API calls or automatic quality verdicts."""
    from src.instruction_data import split_sections, document_title, SKIP_SECTIONS, LLM_PROMPT_TEMPLATE
    if (REVISION/'generation_requests.jsonl').exists():
        raise FileExistsError('Requests already exist; preserve/reconcile your revision before preparing again.')
    requests=[]
    for path in sorted(corpus_dir.glob('*.txt')):
        text=path.read_text(encoding='utf-8');title=document_title(path,text)
        for i,(heading,lines) in enumerate(split_sections(text)):
            body=' '.join(lines)
            if SKIP_SECTIONS.search(heading) or len(body.split()) < 40: continue
            # Keep complete sections together; use these in a chosen LLM or manual editing.
            requests.append({'source':path.name,'group':f'{path.stem}::{i}', 'section':heading,
                             'document_title':title,'prompt':LLM_PROMPT_TEMPLATE.format(n=3,title=title,chunk=body)})
    if not requests: raise ValueError('No usable evidence sections found.')
    write_jsonl(REVISION/'generation_requests.jsonl',requests)
    (REVISION/'generation_prompt.txt').write_text(LLM_PROMPT_TEMPLATE,encoding='utf-8')
    print(f'Prepared {len(requests)} evidence requests from {len(set(r["source"] for r in requests))} sources.')
    print('Save generated or manually paraphrased rows to instruction_revision/candidates.jsonl; retain source/group and evidence_quote.')


def validate(corpus_dir=config.DOMAIN_CORPUS_DIR):
    corpus={p.name:p.read_text(encoding='utf-8') for p in corpus_dir.glob('*.txt')}
    rows=read_jsonl(REVISION/'candidates.jsonl');accepted=[];rejected=[];questions=set();answers=set()
    for row in rows:
        errors=check_pair(row,corpus)
        q=normalized(row.get('instruction',''));a=normalized(row.get('response',''))
        if q in questions or a in answers: errors.append('duplicate_question_or_answer')
        if errors: rejected.append({'row':row,'errors':errors});continue
        questions.add(q);answers.add(a)
        row={**row,'id':hashlib.sha256((q+'\n'+a).encode()).hexdigest()[:16]}
        accepted.append(row)
    templates=Counter(r['template_family'] for r in accepted)
    prefixes=Counter(' '.join(r['instruction'].lower().split()[:2]) for r in accepted)
    types=Counter(r['question_type'] for r in accepted)
    missing=sorted(set(corpus)-{r['source'] for r in accepted})
    coverage={s:dict(Counter(r['question_type'] for r in accepted if r['source']==s)) for s in corpus}
    report={'candidate_count':len(rows),'accepted_count':len(accepted),'rejected_count':len(rejected),
            'dataset_fingerprint':fingerprint(accepted),'minimum_50_pairs_met':len(accepted)>=50,
            'question_types':dict(types),'source_type_coverage':coverage,'missing_sources':missing,
            'question_opening_counts':dict(prefixes),
            'question_openings_over_20_percent':{t:round(n/len(accepted),4) for t,n in prefixes.items() if n/len(accepted)>0.2},
            'templates':dict(templates),'templates_over_20_percent':{t:round(n/len(accepted),4) for t,n in templates.items() if n/len(accepted)>0.2},
            'automated_checks_only':True,'manual_review_required':True}
    write_jsonl(REVISION/'validated_candidates.jsonl',accepted)
    write_jsonl(REVISION/'rejected_candidates.jsonl',rejected)
    (REVISION/'quality_report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Do not overwrite completed reviews on an unchanged dataset.
    audit=REVISION/'random_audit.json'
    if audit.exists() and json.loads(audit.read_text())['dataset_fingerprint'] != fingerprint(accepted):
        audit.rename(REVISION/f'audit_superseded_{hashlib.sha256(audit.read_bytes()).hexdigest()[:12]}.json')
    if not audit.exists():
        sample=random.Random(config.SEED).sample(accepted,min(15,len(accepted)))
        audit.write_text(json.dumps({'dataset_fingerprint':fingerprint(accepted),'seed':config.SEED,
          'reviewer':'','rows':[{**r,**{f:'pending' for f in REVIEW_FIELDS},'notes':''} for r in sample]},indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


def publish(corpus_dir=config.DOMAIN_CORPUS_DIR, output_dir=config.INSTRUCTION_DIR):
    from src.instruction_data import grouped_split
    report=validate(corpus_dir)
    if not report['minimum_50_pairs_met'] or report['missing_sources'] or set(report['question_types']) != TYPES:
        raise ValueError('Need >=50 passing pairs, all corpus sources, and all three question types.')
    audit=json.loads((REVISION/'random_audit.json').read_text())
    if len(audit['rows'])!=15 or not audit['reviewer'].strip() or any(r[f]!='pass' for r in audit['rows'] for f in REVIEW_FIELDS):
        raise ValueError('Complete the seeded 15-pair audit with reviewer name, notes, and pass/fail verdicts; revise failed pairs.')
    if any(not r.get('notes','').strip() for r in audit['rows']): raise ValueError('Add evidence-based audit notes for each pair.')
    if report['templates_over_20_percent']:
        print('WARNING: template families exceed 20%; inspect the flagged families and explain or rebalance them.')
    rows=read_jsonl(REVISION/'validated_candidates.jsonl')
    expected=random.Random(config.SEED).sample(rows,15)
    for reviewed, original in zip(audit['rows'],expected):
        if any(reviewed.get(k)!=v for k,v in original.items()): raise ValueError('Audit content differs from validated sample.')
    pages=list(csv.DictReader((config.REPORT_DIR/'extraction_report.csv').open()))
    # Count only source documents retained in the cleaned corpus, not planted duplicates.
    names={p.stem for p in corpus_dir.glob('*.txt')}
    page_count=sum(int(r['total_pages']) for r in pages if Path(r['file_name']).stem in names)
    if page_count<300: raise ValueError(f'Only {page_count} pages linked to retained corpus; >=300 required.')
    train,evaluation=grouped_split(rows)
    if not train or not evaluation: raise ValueError('Both train and evaluation sets must be nonempty.')
    assert not ({r['group'] for r in train}&{r['group'] for r in evaluation})
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in [('instruction_dataset',rows),('instruction_train',train),('instruction_eval',evaluation)]:
        path=output_dir/f'{name}.jsonl'
        if path.exists():
            old=path.read_bytes();backup=REVISION/'previous_datasets'/hashlib.sha256(old).hexdigest()[:12]/path.name
            backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(old)
        write_jsonl(path,data)
    manifest={**report,'method':'reviewed_paraphrases','total_pairs':len(rows),'train_pairs':len(train),
              'eval_pairs':len(evaluation),'pages_in_retained_corpus':page_count,'audit_reviewer':audit['reviewer'],
              'automated_checks_only':False, 'audit_review_kind':audit.get('review_kind','unspecified'),
              'human_review_completed':audit.get('human_review_completed',False),
              'split_unit':'section group; exact normalized response duplicates removed',
              'files':{name:hashlib.sha256((output_dir/name).read_bytes()).hexdigest() for name in ['instruction_dataset.jsonl','instruction_train.jsonl','instruction_eval.jsonl']}}
    (output_dir/'quality_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (config.REPORT_DIR/'instruction_dataset_stats.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Published reviewed dataset. Previous adapter/results are stale; rerun SFT and evaluation.')
    return manifest


def verify_published():
    manifest=json.loads((config.INSTRUCTION_DIR/'quality_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((config.INSTRUCTION_DIR/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Dataset changed after quality validation; republish before training.')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','validate','publish'])
    args=parser.parse_args();globals()[args.action]()

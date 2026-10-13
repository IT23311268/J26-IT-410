import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'backend')
from services.reference_verifier import extract_references_from_pdf, verify_reference

main_pdf = 'backend/uploads/framework-for-hallucination-detection-in-large-language-models-IJERTV15IS043963_36dff29d.pdf'
refs = extract_references_from_pdf(main_pdf)
print(f'\n=== Parser count: {len(refs)} ===\n')

# Target refs to trace (the 8 known-real papers from the task)
TARGET = {2, 5, 13, 16, 17, 19, 20, 22}

results = {}
for i, ref in enumerate(refs, 1):
    if i in TARGET:
        print(f'\n{"="*60}')
        print(f'=== R{i}: {ref[:100]} ===')
        print(f'{"="*60}')
        v = verify_reference(ref, ref_index=i)
        results[i] = v
        status = v['status']
        score = v['score']
        src = v['matched_source'] or 'None'
        title = (v['matched_title'] or '')[:60]
        print(f'\n>>> RESULT: status={status}  score={score:.3f}  source={src}')
        print(f'>>> matched_title={title}')

print('\n\n========== REGRESSION SUMMARY ==========')
print(f'Parser count confirmed: {len(refs)} (expected 28)')
print()
for i in sorted(TARGET):
    v = results.get(i, {})
    ref_text = refs[i-1] if i <= len(refs) else 'N/A'
    status = v.get('status', 'MISSING')
    score = v.get('score', 0.0)
    src = v.get('matched_source', 'N/A')
    expected = 'REAL'
    ok = 'OK' if status == expected else 'CHANGED'
    print(f'R{i:2d}  [{status:10s}]  score={score:.3f}  src={src}')

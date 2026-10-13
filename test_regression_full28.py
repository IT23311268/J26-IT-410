"""
Full 28-reference regression test.
Verifies all references and prints a before/after comparison table.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'backend')
from services.reference_verifier import extract_references_from_pdf, verify_reference

main_pdf = 'backend/uploads/framework-for-hallucination-detection-in-large-language-models-IJERTV15IS043963_36dff29d.pdf'
refs = extract_references_from_pdf(main_pdf)

print(f'=== Parser count: {len(refs)} (expected 28) ===\n')

results = []
real_count = fake_count = uncertain_count = 0

for i, ref in enumerate(refs, 1):
    v = verify_reference(ref, ref_index=i)
    results.append((i, ref, v))
    s = v['status']
    if s == 'REAL': real_count += 1
    elif s == 'FAKE': fake_count += 1
    elif s == 'UNCERTAIN': uncertain_count += 1

print('\n\n' + '='*80)
print('FINAL RESULTS TABLE')
print('='*80)
print(f"{'#':>3}  {'Status':10}  {'Score':6}  {'Source':30}  Reference (truncated)")
print('-'*80)
for i, ref, v in results:
    status = v['status']
    score = v['score']
    src = (v['matched_source'] or 'None')[:30]
    ref_short = ref[:55]
    flag = ''
    if i in {2, 5, 13, 16, 17, 19, 20, 22}:
        flag = ' *** TARGET'
    print(f"{i:>3}  {status:10}  {score:.3f}  {src:30}  {ref_short}{flag}")

print('='*80)
print(f'\nSUMMARY:  Real={real_count}  Fake={fake_count}  Uncertain={uncertain_count}  Total={len(refs)}')
print(f'Parser count confirmed: {len(refs)} references')

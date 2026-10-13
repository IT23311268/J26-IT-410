import sys

with open('backend/services/analysis_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace all print( with safe_print(
code = code.replace('print(', 'safe_print(')
# Revert any safe_safe_print if we accidentally ran this twice
code = code.replace('safe_safe_print(', 'safe_print(')

prefix = """
def safe_print(*args, **kwargs):
    try:
        import sys
        if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
            msg = " ".join(str(a) for a in args)
            sys.stdout.buffer.write((msg + "\\n").encode("utf-8", "replace"))
        else:
            # We must use builtins.print to avoid recursion if we overrode print
            import builtins
            builtins.print(*args, **kwargs)
    except Exception:
        pass
"""

if "def safe_print" not in code:
    code = code.replace('import logging', 'import logging\n' + prefix)

with open('backend/services/analysis_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Patched analysis_pipeline.py successfully!')

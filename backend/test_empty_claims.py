import sys
sys.path.insert(0, ".")
from services.analysis_pipeline import run_analysis_pipeline
import traceback

text = """
"SummEval: Re-evaluating summarization evaluation," Transactions of the Association
for Computational Linguistics, vol. 9, pp. 391-409, 2021. [Online]. Available:
https://aclanthology.org/2021.tacl-1.24/
[12] O. Honovich, R. Aharoni, J. Herzig, H. Taitelbaum, D. Kukliansy, V. Cohen, T. Scialom,
I. Szpektor, A. Hassidim, and Y. Matias, "TRUE: Re-evaluating factual consistency evalua
tion," 2022.
[13] Z. Xue, R. Li, Q. Dai, and Z. Jiang, "CorefDRE: Document-level relation extraction with
coreference resolution," 2022.
"""

try:
    res = run_analysis_pipeline(text, "uploads")
    print(res)
except Exception as e:
    traceback.print_exc()

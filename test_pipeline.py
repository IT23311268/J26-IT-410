import sys
sys.path.insert(0, "backend")

import asyncio
from services.analysis_pipeline import run_analysis_pipeline

async def main():
    try:
        res = run_analysis_pipeline("req-123", "test_lr", ["backend/uploads/Graphing_the_Truth_Structured_Visualizations_for_A_6cddbe09.pdf"])
        print("Success!")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(main())

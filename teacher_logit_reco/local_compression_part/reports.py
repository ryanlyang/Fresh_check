"""Final reporting utilities for local-compression ParT runs.

Step 14 will compare baseline, MLP delta, local compression, context-only,
random grouping, and optional larger-ParT controls here.
"""

from __future__ import annotations

from .config import LOCAL_COMPRESSION_PART_CONTRACT


LOCAL_COMPRESSION_REPORT_STEP = "local_compression_part_step14_reports"
LOCAL_COMPRESSION_REPORT_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_reports_pending"

"""One-tap hairline preview pipeline.

prepare -> face landmarks -> hair-region mask -> edit model -> align + composite -> return.
"""
from .pipeline import PipelineResult, PreviewError, run_preview

__all__ = ["PipelineResult", "PreviewError", "run_preview"]

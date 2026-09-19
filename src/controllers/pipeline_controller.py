"""
AlphaPulse pipeline controller.

Provides the delivery-layer interface for scheduled and manual
pipeline execution.

The controller delegates orchestration to the existing daily
market pipeline.
"""

from __future__ import annotations

from src.pipelines.daily_market_pipeline import (
    run_daily_pipeline,
)


def run_daily_market_update() -> dict:
    """
    Execute the AlphaPulse daily market pipeline.
    """

    return run_daily_pipeline()
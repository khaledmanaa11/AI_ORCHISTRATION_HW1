"""Public data types returned by NeuralSignalSDK."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from neural_signal.services.evaluator import EvalResult


@dataclass
class RunResult:
    """Aggregated result of training and evaluating all three models."""

    fcn_eval: EvalResult
    rnn_eval: EvalResult
    lstm_eval: EvalResult
    comparison_df: pd.DataFrame

"""Unified callback system — ONE import, no adapters.

Callbacks let you hook into training and optimisation loops.

Two families, one import
------------------------

**Agent training** (epoch-based, ``agent.compile()/fit()``)::

    from agentomatic.callbacks import TrainingEarlyStopping
    cb = TrainingEarlyStopping(monitor="loss", patience=3)
    agent.compile(callbacks=[cb])

**Prompt optimisation** (round-based, ``PromptFitter``) — **no adapter needed**::

    from agentomatic.callbacks import OptimizeEarlyStopping, ModelCheckpoint
    from agentomatic.optimize import PromptFitter

    fitter = PromptFitter(...,
        callbacks=[OptimizeEarlyStopping(patience=3), ModelCheckpoint()],
    )

Reference
---------

======================  ================================  ========================
Class                   Family                            Purpose
======================  ================================  ========================
TrainingCallback        Training (agent.compile/fit)      Base class
TrainingEarlyStopping   Training                          Stop when metric stagnates
OptimizeCallback        Optimisation (PromptFitter)        Base class
OptimizeEarlyStopping   Optimisation                      Stop when score stagnates
ModelCheckpoint         Optimisation                      Save best prompt to disk
NaNStopping             Optimisation                      Halt on NaN scores
PlateauStopping         Optimisation                      Reduce temperature
ScoreThreshold          Optimisation                      Stop at target score
TemperatureScheduler    Optimisation                      Anneal temperature
ProgressLogger          Optimisation                      Log per-round progress
default_callbacks()     Optimisation                      Sensible default stack
======================  ================================  ========================
"""

from __future__ import annotations

from agentomatic.agents.history import Callback as _TrainingCallback
from agentomatic.agents.history import EarlyStopping as _TrainingEarlyStopping
from agentomatic.agents.history import EpochDiffCallback as _EpochDiffCallback
from agentomatic.optimize.callbacks import Callback as _OptimizeCallback
from agentomatic.optimize.callbacks import CallbackContext
from agentomatic.optimize.callbacks import EarlyStopping as _OptimizeEarlyStopping
from agentomatic.optimize.callbacks import ModelCheckpoint as _ModelCheckpoint
from agentomatic.optimize.callbacks import NaNStopping as _NaNStopping
from agentomatic.optimize.callbacks import PlateauStopping as _PlateauStopping
from agentomatic.optimize.callbacks import ProgressLogger as _ProgressLogger
from agentomatic.optimize.callbacks import ScoreThreshold as _ScoreThreshold
from agentomatic.optimize.callbacks import TemperatureScheduler as _TemperatureScheduler
from agentomatic.optimize.callbacks import default_callbacks as _default_callbacks

TrainingCallback = _TrainingCallback
TrainingEarlyStopping = _TrainingEarlyStopping
EpochDiffCallback = _EpochDiffCallback
OptimizeCallback = _OptimizeCallback
OptimizeEarlyStopping = _OptimizeEarlyStopping
ModelCheckpoint = _ModelCheckpoint
NaNStopping = _NaNStopping
PlateauStopping = _PlateauStopping
ProgressLogger = _ProgressLogger
ScoreThreshold = _ScoreThreshold
TemperatureScheduler = _TemperatureScheduler
default_callbacks = _default_callbacks

__all__ = [
    "TrainingCallback",
    "TrainingEarlyStopping",
    "EpochDiffCallback",
    "OptimizeCallback",
    "OptimizeEarlyStopping",
    "ModelCheckpoint",
    "NaNStopping",
    "PlateauStopping",
    "ProgressLogger",
    "ScoreThreshold",
    "TemperatureScheduler",
    "default_callbacks",
    "CallbackContext",
]

"""ClimateNet-Bench model zoo.

Provides a common interface for all models via :class:`ClimateModel`:

- **Baselines**: :class:`ClimatologyBaseline`, :class:`PersistenceBaseline`
- **Linear**: :class:`LinearRegressionModel`
- **Tree ensembles**: :class:`RandomForestModel`, :class:`XGBoostModel`, :class:`LightGBMModel`
- **Deep learning**: :class:`TCNRegressor` (requires 3D sequence arrays)

Use :func:`create_model` from :mod:`climatenet.models.model_factory` to
instantiate models by name.
"""

from climatenet.models.base import ClimateModel
from climatenet.models.climatology import ClimatologyBaseline
from climatenet.models.linear import LinearRegressionModel
from climatenet.models.model_factory import create_model, create_tcn_model, list_available_models
from climatenet.models.persistence import PersistenceBaseline
from climatenet.models.tcn import TCNRegressor
from climatenet.models.tree_models import LightGBMModel, RandomForestModel, XGBoostModel

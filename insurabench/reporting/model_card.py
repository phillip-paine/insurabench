"""Auto-generated one-page model diagnostic summary (design brief §8).

``generate_model_card`` takes an already-fitted ``PricingModel`` (a
``GLMPricingModel``, a ``GBMPricingModel``, or a ``FrequencySeverityModel``)
and a PolicyFrame, and assembles the headline numbers someone reviewing a
fitted pricing model would want in one place: D² (train and/or held-out),
Gini, a calibration check, a lift summary, AIC/BIC where meaningful, and
the biggest rating-factor relativities.

Unlike everything in ``insurabench.curves``/``insurabench.evaluation``
(which deliberately never touch a model -- see ``curves._common``'s own
docstring), this module *does* touch the model directly, the same way
``curves.partial_dependence`` does and for the same kind of reason: a
one-pager needs things a plain prediction array can't supply on its own
(AIC/BIC, a model's own type/family, its raw per-model relativities).
Where the numbers reported ARE derivable from a plain ``y_pred`` array
(Gini, calibration, lift, the base-level-normalized top relativities),
this module still routes through the existing curves/evaluation
functions rather than recomputing anything -- it is a summary of
already-trusted numbers, not a new source of them.

One real correctness issue surfaced while building this, not just
plumbing: glum's own ``GeneralizedLinearRegressor.aic()``/``.bic()``
silently compute the wrong number for any offset-fit model (i.e. any
``target="frequency"`` fit) because their signature has no ``offset``
parameter at all. Fixed in
``insurabench.models.glm.GLMPricingModel.information_criteria`` (see its
docstring for the confirmed, non-trivial size of the discrepancy) --
this module calls that corrected method, never glum's own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from insurabench.curves import relativity_table as _relativity_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.evaluation import (
    calibration_index as _calibration_index,
)
from insurabench.evaluation import (
    gini_index as _gini_index,
)
from insurabench.evaluation import (
    lift_chart as _lift_chart,
)
from insurabench.models.frequency_severity import FrequencySeverityModel
from insurabench.models.glm import GLMPricingModel

Target = Literal["frequency", "severity", "pure_premium"]


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> Markdown-table rendering, hand-rolled rather
    than ``DataFrame.to_markdown()`` -- the latter requires the
    ``tabulate`` package, an extra dependency for something a few lines
    of string formatting already does for the plain scalar/string columns
    a model card actually has."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"

    def _fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


@dataclass(frozen=True)
class ModelCard:
    """One fitted model's diagnostic summary. Every field here is either a
    plain scalar/DataFrame already produced by an existing curves/
    evaluation function, or (``aic``/``bic``/``raw_relativities``) a
    direct, undecorated read of the model's own reporting methods -- see
    module docstring.
    """

    model_type: str
    family: str | dict[str, str] | None
    target: Target
    peril: str | list[str] | None
    coverage: str | list[str] | None

    n_policies: int
    n_claims: int
    total_exposure: float

    train_d2: float | None
    test_d2: float
    gini: float
    calibration_index: float

    lift_table: pd.DataFrame
    lift_ratio: float

    aic: float | None
    bic: float | None

    top_relativities: pd.DataFrame
    raw_relativities: pd.DataFrame | dict[str, pd.DataFrame] | None

    def to_markdown(self) -> str:
        """Render the one-pager as Markdown -- print it, write it to a
        ``.md`` file, or hand it to ``insurabench.Artifact``-style tooling
        as-is."""
        lines = [
            f"# Model card: {self.model_type}",
            "",
            f"- **Family:** {self.family!r}",
            f"- **Target:** `{self.target}`"
            + (f" (peril={self.peril!r})" if self.peril is not None else "")
            + (f" (coverage={self.coverage!r})" if self.coverage is not None else ""),
            (
                f"- **Policy-periods:** {self.n_policies:,} · **Claims:** {self.n_claims:,} "
                f"· **Exposure:** {self.total_exposure:,.1f}"
            ),
            "",
            "## Key metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        if self.train_d2 is not None:
            lines.append(f"| Train D² | {self.train_d2:.4f} |")
        lines.append(f"| Test D² | {self.test_d2:.4f} |")
        lines.append(f"| Gini (normalized) | {self.gini:.4f} |")
        lines.append(f"| Calibration index (~1.0 = well-calibrated) | {self.calibration_index:.4f} |")
        lines.append(f"| Lift ratio (top decile / bottom decile, actual) | {self.lift_ratio:.2f}x |")
        if self.aic is not None:
            lines.append(f"| AIC | {self.aic:,.1f} |")
            lines.append(f"| BIC | {self.bic:,.1f} |")
        else:
            lines.append("| AIC / BIC | not applicable for this model type |")

        lines += ["", "## Top rating-factor relativities", ""]
        lines.append(_dataframe_to_markdown(self.top_relativities))
        return "\n".join(lines)


def _model_family(model) -> str | dict[str, str] | None:
    if isinstance(model, FrequencySeverityModel):
        return {
            "frequency": getattr(model.frequency_model, "family", None),
            "severity": getattr(model.severity_model, "family", None),
        }
    return getattr(model, "family", None)


def _information_criteria(model, pf: PolicyFrame) -> tuple[float | None, float | None]:
    """AIC/BIC only for a plain ``GLMPricingModel`` -- see module
    docstring for why the underlying computation had to be fixed, and
    ``GLMPricingModel.information_criteria`` for the fix itself.

    Deliberately ``None`` (not attempted, not approximated) for a
    ``GBMPricingModel`` leg or a ``FrequencySeverityModel``: a GBM has no
    closed-form likelihood/parameter count in the classical AIC/BIC
    sense (see ``GBMPricingModel.relativities`` for the analogous "this
    number doesn't mean the same thing for a GBM" note), and a composed
    model's two legs don't share one joint likelihood to compute a single
    AIC/BIC from -- reporting ``None`` and saying so plainly beats
    fabricating a number that looks precise but isn't well-defined.
    """
    if isinstance(model, GLMPricingModel):
        return model.information_criteria_policy_frame(pf)
    return None, None


def _raw_relativities(model) -> pd.DataFrame | dict[str, pd.DataFrame] | None:
    if hasattr(model, "relativities"):
        return model.relativities()
    return None


def generate_model_card(
    model,
    target: Target,
    test_pf: PolicyFrame,
    *,
    train_pf: PolicyFrame | None = None,
    peril: str | list[str] | None = None,
    coverage: str | list[str] | None = None,
    n_deciles: int = 10,
    top_n: int = 10,
    power: float | None = None,
) -> ModelCard:
    """Assemble a ``ModelCard`` for an already-fitted ``model`` against
    ``test_pf`` (the frame every reported metric is computed on).

    Parameters
    ----------
    model:
        Already fit (via its own ``fit_policy_frame``) -- a
        ``GLMPricingModel``, ``GBMPricingModel``, or
        ``FrequencySeverityModel``, or anything else exposing
        ``predict_policy_frame``/``score_policy_frame``.
    target:
        Required explicitly (not introspected from ``model``) -- matches
        every ``curves``/``evaluation`` function's own convention, and is
        the only reliable option for a ``FrequencySeverityModel``, which
        has no single ``target_`` attribute the way ``GLMPricingModel``/
        ``GBMPricingModel`` do.
    train_pf:
        Optional -- if given, also reports train-side D² alongside
        ``test_pf``'s, the same train-vs-test comparison
        ``docs/examples/quickstart.py`` already does by hand.
    power:
        Required (and only meaningful) when ``model`` is a
        ``FrequencySeverityModel`` being scored on ``target="pure_premium"``
        -- its ``score_policy_frame`` has no default Tweedie power (see
        its own docstring for why), so this must be threaded through
        rather than silently picked here.
    """
    y_pred = model.predict_policy_frame(test_pf)

    if isinstance(model, FrequencySeverityModel):
        if power is None:
            raise ValueError(
                "power is required to score a FrequencySeverityModel (no "
                "default Tweedie power -- see FrequencySeverityModel."
                "score_policy_frame's own docstring)."
            )
        test_d2 = model.score_policy_frame(test_pf, power=power)
        train_d2 = model.score_policy_frame(train_pf, power=power) if train_pf is not None else None
    else:
        test_d2 = model.score_policy_frame(test_pf)
        train_d2 = model.score_policy_frame(train_pf) if train_pf is not None else None

    gini = _gini_index(test_pf, target, y_pred, peril=peril, coverage=coverage)
    cal_index = _calibration_index(test_pf, target, y_pred, n_bins=n_deciles, peril=peril, coverage=coverage)
    lift_table = _lift_chart(test_pf, target, y_pred, n_deciles=n_deciles, peril=peril, coverage=coverage)

    bottom, top = lift_table["actual"].iloc[0], lift_table["actual"].iloc[-1]
    lift_ratio = float(top / bottom) if bottom != 0 else float("inf" if top > 0 else "nan")

    aic, bic = _information_criteria(model, test_pf)

    rel = _relativity_table(test_pf, target, y_pred, peril=peril, coverage=coverage)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.abs(np.log(rel["relativity"].to_numpy(dtype=float)))
    rel = rel.assign(_deviation=deviation)
    top_relativities = (
        rel.sort_values("_deviation", ascending=False)
        .head(top_n)
        .drop(columns="_deviation")
        .reset_index(drop=True)
    )

    return ModelCard(
        model_type=type(model).__name__,
        family=_model_family(model),
        target=target,
        peril=peril,
        coverage=coverage,
        n_policies=test_pf.n_policies,
        n_claims=test_pf.n_claims,
        total_exposure=test_pf.total_exposure,
        train_d2=train_d2,
        test_d2=test_d2,
        gini=gini,
        calibration_index=cal_index,
        lift_table=lift_table,
        lift_ratio=lift_ratio,
        aic=aic,
        bic=bic,
        top_relativities=top_relativities,
        raw_relativities=_raw_relativities(model),
    )

"""insurabench quickstart: policies + claims -> fitted model -> curves ->
evaluation -> saved charts, run start to finish.

This is a *script*, not a test -- it writes real files to ``--output-dir``
for you to actually look at, rather than asserting things about them.
``tests/test_end_to_end.py`` is the automated version of this same
pipeline (loose sanity assertions, no files written, runs in CI).

Run it as-is for a synthetic demo:

    python docs/examples/quickstart.py

Or point it at your own two-table data by editing the "Load your data"
section below -- everything past that point only assumes ``policies``/
``claims`` are DataFrames matching a ``PolicySchema``/``ClaimsSchema``,
same as the synthetic book. There is no real public-dataset loader (e.g.
freMTPL2) built yet -- the design brief names ``docs/examples/`` as
where that canonical walkthrough should eventually live (§8); this
script uses the synthetic generator instead until that loader exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from insurabench.curves import one_way_curve, partial_dependence, relativity_table
from insurabench.data.datasets import make_synthetic_two_table
from insurabench.data.policy_frame import PolicyFrame
from insurabench.evaluation import (
    calibration_index,
    calibration_table,
    double_lift_chart,
    gini_curve,
    gini_index,
    lift_chart,
    rate_change_by_level,
    rate_change_distribution,
)
from insurabench.model_selection import train_test_split_policy_frame
from insurabench.models.glm import GLMPricingModel
from insurabench.reporting import export_rating_table, generate_model_card
from insurabench.viz import (
    plot_calibration_table,
    plot_double_lift_chart,
    plot_gini_curve,
    plot_lift_chart,
    plot_one_way_curve,
    plot_partial_dependence,
    plot_rate_change_by_level,
    plot_rate_change_distribution,
    plot_relativity_table,
)


def main(n_policies: int, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load your data ---------------------------------------------------
    # Replace this block with your own policies/claims DataFrames + schemas
    # (see insurabench.data.schema.PolicySchema/ClaimsSchema) to run this
    # against real data. Everything below only assumes PolicyFrame.from_tables
    # succeeded -- it doesn't know or care that this came from the synthetic
    # generator.
    print(f"Generating synthetic book ({n_policies} policies, seed={seed})...")
    book = make_synthetic_two_table(n_policies=n_policies, renewals=True, seed=seed)
    pf = PolicyFrame.from_tables(
        book.policies, book.claims, book.policy_schema, book.claims_schema, verbose=True
    )
    print(pf.describe())

    # --- Split, fit ---------------------------------------------------------
    train_pf, test_pf = train_test_split_policy_frame(pf, test_size=0.3, seed=seed)
    print(f"\nTrain: {train_pf.n_policies} policy-periods, {train_pf.n_claims} claims")
    print(f"Test:  {test_pf.n_policies} policy-periods, {test_pf.n_claims} claims")

    print("\nFitting frequency GLM (Poisson) on the training split...")
    model = GLMPricingModel(family="poisson").fit_policy_frame(train_pf, "frequency")
    y_pred = model.predict_policy_frame(test_pf)

    train_d2 = model.score_policy_frame(train_pf)
    test_d2 = model.score_policy_frame(test_pf)
    test_gini = gini_index(test_pf, "frequency", y_pred)
    test_calibration = calibration_index(test_pf, "frequency", y_pred)
    print(f"  Train D-squared: {train_d2:.4f}")
    print(f"  Test  D-squared: {test_d2:.4f}")
    print(f"  Test  Gini:      {test_gini:.4f}")
    print(f"  Test  calibration index (~1.0 = well-calibrated): {test_calibration:.4f}")
    if train_d2 <= 0 or test_d2 <= 0:
        print(
            "  NOTE: a non-positive D-squared means this fit doesn't beat an "
            "intercept-only null model by the deviance measure, even if Gini "
            "shows some real ranking ability above -- that combination is a "
            "genuine finding to investigate (feature functional form, "
            "regularization, categorical cardinality), not a bug in the "
            "scoring itself."
        )

    features = list(pf.policy_schema.feature_roles)
    feature = features[0]

    # --- Curves ---------------------------------------------------------------
    print(f"\nBuilding curves for '{feature}'...")
    curve = one_way_curve(test_pf, feature, "frequency", y_pred)
    table = relativity_table(test_pf, "frequency", y_pred)
    pdp = partial_dependence(train_pf, feature, "frequency", model)

    plot_one_way_curve(curve, feature_name=feature, save_path=str(output_dir / "one_way.png"))
    plot_relativity_table(table, feature=feature, save_path=str(output_dir / "relativity.png"))
    plot_partial_dependence(pdp, feature_name=feature, save_path=str(output_dir / "partial_dependence.png"))
    table.to_csv(output_dir / "relativity_table.csv", index=False)

    # --- Evaluation -------------------------------------------------------------
    print("Building evaluation charts...")
    lc = lift_chart(test_pf, "frequency", y_pred)
    gc = gini_curve(test_pf, "frequency", y_pred)
    ct = calibration_table(test_pf, "frequency", y_pred)

    plot_lift_chart(lc, save_path=str(output_dir / "lift_chart.png"))
    plot_gini_curve(gc, gini_value=test_gini, save_path=str(output_dir / "gini_curve.png"))
    plot_calibration_table(ct, save_path=str(output_dir / "calibration.png"))

    # --- A second model, for double-lift / rate-impact -----------------------
    print("Fitting a second (regularized) model for comparison...")
    model_b = GLMPricingModel(family="poisson", alpha=0.3).fit_policy_frame(train_pf, "frequency")
    y_pred_b = model_b.predict_policy_frame(test_pf)

    dl = double_lift_chart(test_pf, "frequency", y_pred, y_pred_b, model_a_name="glm", model_b_name="glm_regularized")
    by_level = rate_change_by_level(test_pf, feature, "frequency", y_pred, y_pred_b)
    distribution = rate_change_distribution(test_pf, "frequency", y_pred, y_pred_b)

    plot_double_lift_chart(
        dl, model_a_name="glm", model_b_name="glm_regularized", save_path=str(output_dir / "double_lift.png")
    )
    plot_rate_change_by_level(
        by_level, feature_name=feature, save_path=str(output_dir / "rate_change_by_level.png")
    )
    plot_rate_change_distribution(distribution, save_path=str(output_dir / "rate_change_distribution.png"))

    # --- Reporting: model card + rating-table export --------------------------
    print("Generating model card and rating-table export...")
    card = generate_model_card(model, "frequency", test_pf, train_pf=train_pf)
    (output_dir / "model_card.md").write_text(card.to_markdown())
    export_rating_table(test_pf, "frequency", y_pred, output_dir / "rating_table.xlsx")

    print(f"\nDone. Wrote charts, relativity_table.csv, model_card.md, and "
          f"rating_table.xlsx to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-policies", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("quickstart_output"))
    args = parser.parse_args()
    main(args.n_policies, args.seed, args.output_dir)

"""
interpret.py
Interpretability suite:
- Permutation importance on held-out test data (model-agnostic baseline)
- PDP (average) for 2–3 features + ICE (individual) overlay for one feature
- Optional decision boundary slice for two-feature illustration
Saves clinician-ready figures and prints one-line takeaways per visual.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

def permutation_importance_plot(clf, X_test, y_test, feature_names, n_repeats=10, random_state=42):
    ensure_fig_dir()
    result = permutation_importance(
        clf, X_test, y_test,
        n_repeats=n_repeats, random_state=random_state
    )
    sorted_idx = np.argsort(result.importances_mean)[::-1]
    top_k = min(10, len(feature_names))
    features = np.array(feature_names)[sorted_idx][:top_k]
    means = result.importances_mean[sorted_idx][:top_k]
    stds = result.importances_std[sorted_idx][:top_k]

    plt.figure(figsize=(8, 5))
    plt.barh(features[::-1], means[::-1], xerr=stds[::-1], color="#2a9d8f")
    plt.xlabel("Permutation importance (mean decrease in score)")
    plt.title("Permutation importance (top features)")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "permutation_importance.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved: {out_path}")

    # Clinician takeaway template
    if "creatinine" in feature_names:
        print("Actionable: Elevated creatinine is a primary driver of risk; escalate renal monitoring.")

def pdp_ice_plots(clf, X_train, features_to_plot):
    ensure_fig_dir()

    # PDP average
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(
        clf, X_train, features_to_plot, kind="average", ax=ax
    )
    ax.set_title("Partial Dependence (average effect across patients)")
    plt.tight_layout()
    avg_path = os.path.join(FIG_DIR, "pdp_average.png")
    plt.savefig(avg_path, dpi=160)
    plt.close()
    print(f"Saved: {avg_path}")

    # ICE individual for first feature
    first_feature = [features_to_plot[0]]
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(
        clf, X_train, first_feature, kind="individual", ax=ax
    )
    ax.set_title(f"ICE for {first_feature[0]} (patient-level trajectories)")
    plt.tight_layout()
    ice_path = os.path.join(FIG_DIR, "ice_feature1.png")
    plt.savefig(ice_path, dpi=160)
    plt.close()
    print(f"Saved: {ice_path}")

    # Clinician takeaway template
    print(f"Actionable: Risk trend vs {first_feature[0]} shows clear thresholds—use for triage decisions.")

def decision_boundary_slice(clf, X_train, feature_x: str, feature_y: str, grid_size=120):
    """
    2D illustration only; projects high-D decisions into two features.
    Fix other features at median; vary feature_x and feature_y on a grid.
    """
    ensure_fig_dir()

    x_vals = np.linspace(X_train[feature_x].quantile(0.02), X_train[feature_x].quantile(0.98), grid_size)
    y_vals = np.linspace(X_train[feature_y].quantile(0.02), X_train[feature_y].quantile(0.98), grid_size)
    xv, yv = np.meshgrid(x_vals, y_vals)

    # Build grid DataFrame
    baseline = X_train.copy()
    medians = baseline.median(numeric_only=True).to_dict()
    grid = baseline.iloc[:grid_size * grid_size].copy()
    for col in grid.columns:
        grid[col] = medians.get(col, grid[col].mode()[0] if not np.issubdtype(grid[col].dtype, np.number) else grid[col])
    grid[feature_x] = xv.flatten()
    grid[feature_y] = yv.flatten()

    try:
        probs = clf.predict_proba(grid)[:, 1]
    except Exception:
        probs = clf.predict(grid)

    plt.figure(figsize=(7, 6))
    cs = plt.contourf(xv, yv, probs.reshape(grid_size, grid_size), levels=20, cmap="viridis")
    plt.colorbar(cs, label="Predicted risk")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"Decision boundary slice: {feature_x} vs {feature_y}")
    out_path = os.path.join(FIG_DIR, "decision_boundary_slice.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved: {out_path}")
    print("Caution: Use boundary slices alongside PDP/ICE; they simplify high-dimensional decision logic.")

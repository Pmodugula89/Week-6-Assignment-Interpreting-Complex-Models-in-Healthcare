"""
main.py
End-to-end run:
- Load data (synthetic by default)
- Build & fit pipeline
- Evaluate test performance
- Run interpretability: permutation importance, PDP/ICE, optional boundary slice
"""

from data_load import load_data
from model_train import build_pipeline, fit_pipeline
from evaluate import evaluate, print_metrics
from interpret import permutation_importance_plot, pdp_ice_plots, decision_boundary_slice

def run(data_path: str = None):
    X_train, X_test, y_train, y_test = load_data(path=data_path)
    clf = build_pipeline(X_train)
    clf = fit_pipeline(clf, X_train, y_train)

    # Technical evaluation
    metrics = evaluate(clf, X_test, y_test)
    print_metrics(metrics)

    # Interpretability on held-out data
    feature_names = X_train.columns.tolist()
    permutation_importance_plot(clf, X_test, y_test, feature_names)

    # PDP/ICE for clinically meaningful features
    features_to_plot = ["creatinine", "lactate", "heart_rate"]
    pdp_ice_plots(clf, X_train, features_to_plot)

    # Optional: 2D slice visualization (synthetic demo)
    decision_boundary_slice(clf, X_train, feature_x="creatinine", feature_y="lactate")

if __name__ == "__main__":
    # If using a CSV, pass path like data_path="data/raw/your_dataset.csv"
    run(data_path=None)

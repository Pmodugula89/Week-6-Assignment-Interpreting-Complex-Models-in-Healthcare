# Week-6-Assignment-Interpreting-Complex-Models-in-Healthcare
This project demonstrates how to make complex healthcare AI models **interpretable and clinically actionable**. The HealthInsight data science team developed a pipeline that flags at-risk patients and predicts outcomes, but clinicians require explanations they can trust without needing deep machine learning expertise.

****Project Goals**
-	Build a reproducible pipeline using scikit-learn and XGBoost to predict patient risk.
-	Apply interpretability techniques to explain model behavior and support clinical decisions.
-	Generate clinician-ready visuals and insights using permutation importance, PDP, ICE, and decision boundary slices.
-	
**Folder Structure**
 Week-6-Assignment-Interpreting-Complex-Models-in-Healthcare/ ├── src/                      # Python scripts for data, model, evaluation, interpretation │   ├── data_load.py │   ├── model_train.py │   ├── evaluate.py │   ├── interpret.py │   └── main.py ├── figures/                  # Saved interpretability plots ├── data/                     # Optional raw data folder (if using external dataset) │   └── raw/ ├── requirements.txt          # Python dependencies ├── README.md                 # Project overview and instructions ├── .gitignore                # Ignore .venv, pycache, etc. └── .venv/                    # Virtual environment (not tracked)

**Setup Instructions**
### 1. Clone the repository
git clone https://github.com/Pmodugula89/Week-6-Assignment-Interpreting-Complex-Models-in-Healthcare.git
cd cst600-week06-interpretability-healthcare
create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate.bat     # (Windows Command Prompt)

**Install dependecies**
pip install -r requirements.txt
Run the pipeline
python src\main.py
This will:
-	Load synthetic healthcare-like data
-	Train a Gradient Boosting model
-	Evaluate performance (accuracy, precision, recall, F1, ROC-AUC)
-	Generate interpretability visuals in the figures/ folder
-	
**Outputs**
- figures/permutation_importance.png
- figures/pdp_average.png
- figures/ice_feature1.png
- figures/decision_boundary_slice.png
Each figure includes a one-line clinician takeaway printed in the terminal.

**Methods Used**
-	Model: HistGradientBoostingClassifier with preprocessing pipeline (scaling + one-hot encoding)
-	Evaluation: Accuracy, macro precision/recall/F1, ROC-AUC, calibration note
-	Interpretability:
-	Permutation importance (model-agnostic)
-	Partial Dependence Plots (PDP)
-	Individual Conditional Expectation (ICE)
-	Decision boundary slice (optional 2D projection)
-	
**Example Insights**
-	“Creatinine and lactate are dominant risk drivers; monitor renal function and tissue perfusion early.”
-	“Risk rises steeply when creatinine exceeds ~1.5; consider escalation protocols.”
-	“ICE plots show patient-level variability in heart rate; cohort-specific thresholds may be needed.”
-	
**Limitations & Next Steps**
-	PDP may be affected by feature correlations — consider ALE for robustness.
-	Add fairness audits across age/sex cohorts.
-	Calibrate probabilities if using thresholds for decision-making.
-	Extend to temporal models for longitudinal patient monitoring.
-	
**References**
Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv:1702.08608.
Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.
Goldstein, A., et al. (2015). Peeking inside the black box: ICE plots. JCGS, 24(1), 44–65.

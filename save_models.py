from pathlib import Path
import joblib

# Run this script from the project root AFTER the model-training notebook
# has trained the variables: preprocessor, xgb_model, rf_model, ebm_model.

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.joblib")
joblib.dump(rf_model, MODEL_DIR / "random_forest_model.joblib")
joblib.dump(ebm_model, MODEL_DIR / "ebm_model.joblib")

print("Saved:")
for p in sorted(MODEL_DIR.glob("*.joblib")):
    print(" -", p)

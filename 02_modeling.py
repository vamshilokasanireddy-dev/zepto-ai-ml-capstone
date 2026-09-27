from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "titanic.csv")

target = "survived"
classification_features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

X = df[classification_features].copy()
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ]
)


def make_pipeline(estimator):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(max_depth=5, random_state=42)
    ),
    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            oob_score=True,
        )
    ),
}


def evaluate_classifier(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "fitted_model": model,
    }


plt.figure(figsize=(8, 6))
results = [
    evaluate_classifier(name, model)
    for name, model in models.items()
]
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "roc_curves.png")
plt.close()

comparison = pd.DataFrame(
    [
        {
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "auc": r["auc"],
            "confusion_matrix": str(r["confusion_matrix"]),
        }
        for r in results
    ]
)
comparison.to_csv(OUTPUT / "classifier_comparison.csv", index=False)

# Decision tree visualization with transformed feature names.
tree_pipeline = results[1]["fitted_model"]
feature_names = tree_pipeline.named_steps["preprocessor"].get_feature_names_out()

plt.figure(figsize=(22, 12))
plot_tree(
    tree_pipeline.named_steps["model"],
    feature_names=feature_names,
    class_names=["not_survived", "survived"],
    filled=False,
    max_depth=4,
)
plt.title("Decision Tree")
plt.tight_layout()
plt.savefig(OUTPUT / "decision_tree.png")
plt.close()

# Imbalance comparison on the same train/test split.
imbalance_rows = []

def evaluate_variant(name, estimator):
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    return {
        "variant": name,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


imbalance_rows.append(
    evaluate_variant(
        "baseline",
        make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    )
)

imbalance_rows.append(
    evaluate_variant(
        "class_weight_balanced",
        make_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
        ),
    )
)

# SMOTE must operate only after the train/test split and inside the training pipeline.
smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
imbalance_rows.append(
    evaluate_variant("smote_training_only", smote_pipeline)
)

imbalance_df = pd.DataFrame(imbalance_rows)
imbalance_df.to_csv(OUTPUT / "imbalance_comparison.csv", index=False)

# GridSearchCV for Random Forest.
rf_pipeline = make_pipeline(
    RandomForestClassifier(
        oob_score=True,
        random_state=42,
    )
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
oob_score = best_rf.named_steps["model"].oob_score_

grid_result = pd.DataFrame(
    [
        {
            "best_params": str(grid.best_params_),
            "cv_best_f1": grid.best_score_,
            "oob_score": oob_score,
        }
    ]
)
grid_result.to_csv(OUTPUT / "random_forest_gridsearch.csv", index=False)

# Regression: predict fare from other available features.
regression_df = df.copy()
regression_target = regression_df["fare"]

reg_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
]

X_reg = regression_df[reg_features].copy()

for column in X_reg.columns:
    X_reg[column] = X_reg[column].fillna(X_reg[column].median())

reg_target = regression_target.fillna(regression_target.median())

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    reg_target,
    test_size=0.20,
    random_state=42,
)

reg_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)

reg_model.fit(Xr_train, yr_train)
reg_pred = reg_model.predict(Xr_test)

mae = mean_absolute_error(yr_test, reg_pred)
rmse = np.sqrt(mean_squared_error(yr_test, reg_pred))
r2 = r2_score(yr_test, reg_pred)

n = len(yr_test)
p = Xr_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

residuals = yr_test - reg_pred

plt.figure(figsize=(8, 5))
plt.scatter(reg_pred, residuals, alpha=0.65)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")
plt.tight_layout()
plt.savefig(OUTPUT / "fare_residual_plot.png")
plt.close()

regression_result = pd.DataFrame(
    [
        {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2,
        }
    ]
)
regression_result.to_csv(OUTPUT / "regression_metrics.csv", index=False)

# Complete end-to-end pipeline artifact.
# Use the best Random Forest pipeline selected above.
joblib.dump(best_rf, OUTPUT / "model_pipeline.joblib")

# Reload and predict on raw test input to verify the artifact.
loaded_pipeline = joblib.load(OUTPUT / "model_pipeline.joblib")
reload_predictions = loaded_pipeline.predict(X_test.head(5))

validation = pd.DataFrame(
    {
        "actual": y_test.head(5).to_numpy(),
        "reloaded_pipeline_prediction": reload_predictions,
    }
)
validation.to_csv(OUTPUT / "pipeline_reload_validation.csv", index=False)

# Final report.
# A neutral metric-based deployment recommendation is generated from the observed
# classification metrics. The exact values are produced by the user's run.
best_row = comparison.sort_values(["f1", "auc"], ascending=False).iloc[0]
recommendation = (
    f"Based on the observed test-set metrics, {best_row['model']} has the highest "
    f"F1 score ({best_row['f1']:.4f}) among the three classifiers in this run. "
    f"Its accuracy is {best_row['accuracy']:.4f}, precision is {best_row['precision']:.4f}, "
    f"recall is {best_row['recall']:.4f}, and AUC is {best_row['auc']:.4f}. "
    "This recommendation is based on the measured evaluation results; operational "
    "considerations such as latency and maintenance should also be reviewed before deployment."
)

with open(OUTPUT / "modeling_report.txt", "w", encoding="utf-8") as fh:
    fh.write("CLASS BALANCE\n")
    fh.write(y.value_counts(normalize=True).sort_index().to_string())
    fh.write("\n\nCLASSIFIER COMPARISON\n")
    fh.write(comparison.to_string(index=False))
    fh.write("\n\nIMBALANCE COMPARISON\n")
    fh.write(imbalance_df.to_string(index=False))
    fh.write("\n\nGRID SEARCH\n")
    fh.write(grid_result.to_string(index=False))
    fh.write("\n\nREGRESSION\n")
    fh.write(regression_result.to_string(index=False))
    fh.write("\n\nFINAL CLASSIFIER RECOMMENDATION\n")
    fh.write(recommendation)
    fh.write(
        "\n\nHeteroscedasticity conclusion: inspect the saved residual plot. "
        "A clearly widening or narrowing residual spread across fitted values "
        "would indicate heteroscedasticity; a roughly constant random spread "
        "would not."
    )
    fh.write("\n\nPIPELINE RELOAD VALIDATION\n")
    fh.write(validation.to_string(index=False))

print("Classifier comparison:")
print(comparison.to_string(index=False))
print("\nImbalance comparison:")
print(imbalance_df.to_string(index=False))
print("\nRandom Forest GridSearch:")
print(grid_result.to_string(index=False))
print("\nRegression:")
print(regression_result.to_string(index=False))
print("\nFinal classifier recommendation:")
print(recommendation)
print("\nReload validation:")
print(validation.to_string(index=False))

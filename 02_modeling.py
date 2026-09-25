from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "titanic.csv")
df = df.drop(columns=["deck"], errors="ignore")
df = df.dropna(subset=["survived"]).copy()

features = ["pclass","sex","age","sibsp","parch","fare","embarked"]
X = df[features]
y = df["survived"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric = ["pclass","age","sibsp","parch","fare"]
categorical = ["sex","embarked"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test,pred),
        "Precision": precision_score(y_test,pred,zero_division=0),
        "Recall": recall_score(y_test,pred,zero_division=0),
        "F1": f1_score(y_test,pred,zero_division=0),
        "AUC": auc
    })
    cm = confusion_matrix(y_test,pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / f"cm_{name.lower().replace(' ','_')}.png")
    plt.close()
    fpr,tpr,_ = roc_curve(y_test,proba)
    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"roc_{name.lower().replace(' ','_')}.png")
    plt.close()

# Decision tree visualization with feature names/classes.
tree_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=4, random_state=42))
])
tree_pipe.fit(X_train,y_train)
feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
plt.figure(figsize=(20,10))
plot_tree(tree_pipe.named_steps["model"], feature_names=feature_names,
          class_names=["Not survived","Survived"], filled=True, max_depth=4)
plt.tight_layout()
plt.savefig(OUT / "decision_tree.png")
plt.close()

# Imbalance comparison using Logistic Regression.
def fit_eval(estimator, Xtr, ytr):
    pipe = Pipeline([("preprocess", preprocessor),("model",estimator)])
    pipe.fit(Xtr,ytr)
    p = pipe.predict(X_test)
    return {
        "Precision": precision_score(y_test,p,zero_division=0),
        "Recall": recall_score(y_test,p,zero_division=0),
        "F1": f1_score(y_test,p,zero_division=0)
    }

imbalance = []
for label, est, Xtr, ytr in [
    ("Baseline", LogisticRegression(max_iter=1000,random_state=42), X_train,y_train),
    ("class_weight=balanced", LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42), X_train,y_train)
]:
    m = fit_eval(est,Xtr,ytr)
    m["Method"] = label
    imbalance.append(m)

# SMOTE after preprocessing, on training fold only.
Xtr_processed = preprocessor.fit_transform(X_train)
Xte_processed = preprocessor.transform(X_test)
sm = SMOTE(random_state=42)
X_sm,y_sm = sm.fit_resample(Xtr_processed,y_train)
sm_model = LogisticRegression(max_iter=1000,random_state=42)
sm_model.fit(X_sm,y_sm)
sm_pred = sm_model.predict(Xte_processed)
imbalance.append({
    "Method":"SMOTE (train only)",
    "Precision":precision_score(y_test,sm_pred,zero_division=0),
    "Recall":recall_score(y_test,sm_pred,zero_division=0),
    "F1":f1_score(y_test,sm_pred,zero_division=0)
})

pd.DataFrame(imbalance).to_csv(OUT / "imbalance_comparison.csv",index=False)

# Random Forest GridSearch.
rf_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1))
])
param_grid = {
    "model__n_estimators":[100,200],
    "model__max_depth":[None,5,10],
    "model__max_features":["sqrt","log2"]
}
grid = GridSearchCV(rf_pipe,param_grid,cv=3,scoring="f1",n_jobs=-1)
grid.fit(X_train,y_train)
best_pipe = grid.best_estimator_
best_rf = best_pipe.named_steps["model"]

# Regression side task: predict fare from other available features.
reg_features = ["pclass","sex","age","sibsp","parch","embarked","survived"]
reg_df = df[reg_features + ["fare"]].copy()
Xr = reg_df[reg_features]
yr = reg_df["fare"]
Xr_train,Xr_test,yr_train,yr_test = train_test_split(
    Xr,yr,test_size=0.2,random_state=42
)
reg_num = ["pclass","age","sibsp","parch","survived"]
reg_cat = ["sex","embarked"]
reg_pre = ColumnTransformer([
    ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),reg_num),
    ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),reg_cat)
])
reg_pipe = Pipeline([("preprocess",reg_pre),("model",LinearRegression())])
reg_pipe.fit(Xr_train,yr_train)
yr_pred = reg_pipe.predict(Xr_test)
mae = mean_absolute_error(yr_test,yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test,yr_pred))
r2 = r2_score(yr_test,yr_pred)
n,p = len(yr_test), Xr_test.shape[1]
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

resid = yr_test - yr_pred
plt.figure()
plt.scatter(yr_pred,resid)
plt.axhline(0,linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "regression_residuals.png")
plt.close()

comparison = pd.DataFrame(results)
comparison.to_csv(OUT / "classification_comparison.csv",index=False)

with open(OUT / "model_report.txt","w",encoding="utf-8") as f:
    f.write("CLASSIFICATION COMPARISON\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\nIMBALANCE COMPARISON\n")
    f.write(pd.DataFrame(imbalance).to_string(index=False))
    f.write("\n\nGRID SEARCH\n")
    f.write(f"Best params: {grid.best_params_}\n")
    f.write(f"Best CV F1: {grid.best_score_:.4f}\n")
    f.write(f"OOB score of fitted best RF: {best_rf.oob_score_:.4f}\n")
    f.write("\nREGRESSION\n")
    f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\nAdjusted R2: {adj_r2:.4f}\n")
    f.write("Residual interpretation: inspect residual plot; a funnel/structured spread indicates heteroscedasticity, while a roughly random constant-width spread is more consistent with homoscedasticity.\n")

# Save the complete preprocessing + final classifier pipeline.
joblib.dump(best_pipe, ROOT / "model_pipeline.joblib")

# Reload test using raw, unprocessed rows.
loaded = joblib.load(ROOT / "model_pipeline.joblib")
sample_prediction = loaded.predict(X_test.iloc[:3])
with open(OUT / "reload_test.txt","w") as f:
    f.write("Reloaded complete pipeline predictions on raw rows: " + str(sample_prediction))

print(comparison.to_string(index=False))
print("Saved:", ROOT / "model_pipeline.joblib")

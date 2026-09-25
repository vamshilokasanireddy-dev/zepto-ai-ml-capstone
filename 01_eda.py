from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# The raw network/cache load happens exactly once in this module.
df = sns.load_dataset("titanic")
df.to_csv(ROOT / "titanic.csv", index=False)

with open(OUT / "eda_report.txt", "w", encoding="utf-8") as f:
    f.write("INFO\n")
    df.info(buf=f)
    f.write("\nDESCRIBE\n")
    f.write(df.describe(include="all").to_string())
    f.write(f"\n\nSHAPE: {df.shape}\n")
    missing = (df.isna().mean() * 100)
    f.write("\nMISSING PERCENTAGES (affected columns)\n")
    f.write(missing[missing > 0].to_string())

# Required threshold-based cleaning.
# age is in 5-30% range -> median impute.
# embarked is under 5% -> drop affected rows.
# deck is >30% -> drop unreliable column.
clean = df.copy()
age_pct = clean["age"].isna().mean() * 100
embarked_pct = clean["embarked"].isna().mean() * 100
deck_pct = clean["deck"].isna().mean() * 100

clean["age"] = clean["age"].fillna(clean["age"].median())
clean = clean.dropna(subset=["embarked"]).copy()
clean = clean.drop(columns=["deck"])
clean = clean.dropna(subset=["fare", "sex", "pclass", "survived"])

with open(OUT / "eda_report.txt", "a", encoding="utf-8") as f:
    f.write(f"\n\nCleaning decisions:\n")
    f.write(f"age missing={age_pct:.2f}% -> median imputation because 5%-30%.\n")
    f.write(f"embarked missing={embarked_pct:.2f}% -> drop affected rows because under 5%.\n")
    f.write(f"deck missing={deck_pct:.2f}% -> drop column because over 30%.\n")

def outlier_count(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())

age_out = outlier_count(clean["age"])
fare_out = outlier_count(clean["fare"])
fare_mean, fare_median, fare_mode = clean["fare"].mean(), clean["fare"].median(), clean["fare"].mode().iloc[0]

# Univariate plots
for col in ["age", "fare"]:
    plt.figure()
    sns.histplot(clean[col], kde=True)
    plt.title(f"{col.title()} Histogram")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_hist.png")
    plt.close()

    plt.figure()
    sns.boxplot(x=clean[col])
    plt.title(f"{col.title()} Box Plot")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_box.png")
    plt.close()

# Bivariate survival rates
sex_survival = clean.groupby("sex")["survived"].mean()
pclass_survival = clean.groupby("pclass")["survived"].mean()
sex_class_survival = clean.groupby(["sex","pclass"])["survived"].mean()

corr_cols = ["survived","pclass","age","sibsp","parch","fare"]
corr = clean[corr_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Required 6x6 Correlation Matrix")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png")
plt.close()

pairs = []
for i in range(len(corr_cols)):
    for j in range(i+1, len(corr_cols)):
        pairs.append((abs(corr.iloc[i,j]), corr.iloc[i,j], corr_cols[i], corr_cols[j]))
pairs.sort(reverse=True)
top2 = pairs[:2]

# Four multivariate charts
charts = [
    ("survival_by_sex.png", clean.groupby("sex")["survived"].mean(), "Survival Rate by Sex"),
    ("survival_by_class.png", clean.groupby("pclass")["survived"].mean(), "Survival Rate by Passenger Class"),
]
for filename, series, title in charts:
    plt.figure()
    series.plot(kind="bar")
    plt.ylabel("Survival rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / filename)
    plt.close()

plt.figure()
sns.boxplot(data=clean, x="survived", y="age")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig(OUT / "age_by_survival.png")
plt.close()

plt.figure()
sns.boxplot(data=clean, x="survived", y="fare")
plt.title("Fare Distribution by Survival")
plt.tight_layout()
plt.savefig(OUT / "fare_by_survival.png")
plt.close()

# Exploratory standardization, separate from modeling pipeline.
scaler = StandardScaler()
z = scaler.fit_transform(clean[["age","fare"]])
zdf = pd.DataFrame(z, columns=["age_z","fare_z"])
before = clean[["age","fare"]].agg(["mean","std"])
after = zdf.agg(["mean","std"])

with open(OUT / "eda_report.txt", "a", encoding="utf-8") as f:
    f.write("\n\nIQR OUTLIERS\n")
    f.write(f"age: {age_out}\nfare: {fare_out}\n")
    f.write("\nFARE DISTRIBUTION\n")
    f.write(f"mean={fare_mean:.4f}, median={fare_median:.4f}, mode={fare_mode:.4f}\n")
    f.write("Interpretation: fare is right-skewed when mean > median > mode.\n")
    f.write("\nSURVIVAL BY SEX\n" + sex_survival.to_string() + "\n")
    f.write("\nSURVIVAL BY PCLASS\n" + pclass_survival.to_string() + "\n")
    f.write("\nSURVIVAL BY SEX AND PCLASS\n" + sex_class_survival.to_string() + "\n")
    f.write("\nCORRELATION MATRIX\n" + corr.to_string() + "\n")
    f.write("\nTWO STRONGEST ABSOLUTE OFF-DIAGONAL CORRELATIONS\n")
    for a,b,c,d in top2:
        f.write(f"{c} vs {d}: correlation={b:.4f}\n")
    f.write("\nCHART INTERPRETATIONS\n")
    f.write("1. Survival by sex: the bars show the observed survival-rate difference between female and male passengers.\n")
    f.write("2. Survival by class: the chart shows how observed survival rate varies across passenger classes.\n")
    f.write("3. Age by survival: the box plots compare age distributions between survivors and non-survivors.\n")
    f.write("4. Fare by survival: the box plots compare fare distributions between survivors and non-survivors.\n")
    f.write("\nSTANDARDIZATION BEFORE\n" + before.to_string())
    f.write("\n\nSTANDARDIZATION AFTER\n" + after.to_string())

clean.to_csv(OUT / "cleaned_titanic.csv", index=False)
print("EDA complete. Offline fallback saved to", ROOT / "titanic.csv")

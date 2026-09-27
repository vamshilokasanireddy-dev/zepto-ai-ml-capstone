from typing import Tuple
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# Required single network/cache load for this module.
df = sns.load_dataset("titanic")

# Required offline fallback.
df.to_csv(ROOT / "titanic.csv", index=False)

report = []

report.append("DATASET PROFILE")
report.append(f"shape = {df.shape}")
report.append("\ndf.info():")
df.info(buf=None)
report.append("\ndf.describe():")
report.append(df.describe(include="all").to_string())

missing = df.isna().mean().mul(100)
missing = missing[missing > 0]
report.append("\nMISSING PERCENTAGES:")
report.append(missing.to_string())

# Apply the requested threshold strategy.
clean = df.copy()

for column in missing.index:
    rate = missing[column]
    if rate < 5:
        clean = clean.dropna(subset=[column])
        strategy = f"drop rows ({rate:.2f}% missing < 5%)"
    elif rate <= 30:
        if pd.api.types.is_numeric_dtype(clean[column]):
            clean[column] = clean[column].fillna(clean[column].median())
        else:
            clean[column] = clean[column].fillna(clean[column].mode().iloc[0])
        strategy = f"impute ({rate:.2f}% missing, between 5% and 30%)"
    else:
        # For this dataset, a high-missing categorical field can be represented
        # as an explicit category rather than silently dropping observations.
        if pd.api.types.is_numeric_dtype(clean[column]):
            clean[column] = clean[column].fillna(clean[column].median())
            strategy = f"median imputation ({rate:.2f}% missing; documented as a high-missing fallback)"
        else:
            clean[column] = clean[column].fillna("Missing")
            strategy = f"encode as 'Missing' ({rate:.2f}% missing > 30%)"
    report.append(f"{column}: {strategy}")

# Univariate statistics and plots.
for column in ["age", "fare"]:
    plt.figure(figsize=(7, 4))
    sns.histplot(clean[column], kde=True)
    plt.title(f"Histogram of {column}")
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{column}_histogram.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(x=clean[column])
    plt.title(f"Box plot of {column}")
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{column}_boxplot.png")
    plt.close()


def iqr_outliers(series: pd.Series) -> Tuple[int, float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return count, lower, upper


age_outliers, age_lower, age_upper = iqr_outliers(clean["age"])
fare_outliers, fare_lower, fare_upper = iqr_outliers(clean["fare"])

fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode().iloc[0]

if fare_mean > fare_median > fare_mode:
    skew_text = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    skew_text = "left-skewed"
else:
    skew_text = "not conclusively skewed from mean/median/mode ordering alone"

report.append("\nIQR OUTLIERS")
report.append(f"age: {age_outliers} (bounds {age_lower:.3f}, {age_upper:.3f})")
report.append(f"fare: {fare_outliers} (bounds {fare_lower:.3f}, {fare_upper:.3f})")
report.append("\nFARE")
report.append(f"mean={fare_mean:.4f}")
report.append(f"median={fare_median:.4f}")
report.append(f"mode={fare_mode:.4f}")
report.append(f"distribution conclusion={skew_text}")

# Bivariate survival rates.
survival_by_sex = clean.groupby("sex")["survived"].mean().mul(100)
survival_by_class = clean.groupby("pclass")["survived"].mean().mul(100)
survival_by_sex_class = (
    clean.groupby(["sex", "pclass"])["survived"].mean().mul(100)
)

report.append("\nSURVIVAL RATE BY SEX (%)")
report.append(survival_by_sex.to_string())
report.append("\nSURVIVAL RATE BY PCLASS (%)")
report.append(survival_by_class.to_string())
report.append("\nSURVIVAL RATE BY SEX AND PCLASS (%)")
report.append(survival_by_sex_class.to_string())

# Explicit boolean masking example.
female_first_or_second = clean[
    (clean["sex"] == "female") & (clean["pclass"].isin([1, 2]))
]
report.append(
    f"\nBoolean-mask example rows (female and pclass 1/2): {len(female_first_or_second)}"
)

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Titanic Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT / "correlation_heatmap.png")
plt.close()

pairs = []
for i, a in enumerate(corr_cols):
    for b in corr_cols[i + 1:]:
        pairs.append((a, b, corr.loc[a, b], abs(corr.loc[a, b])))

top_two = sorted(pairs, key=lambda x: x[3], reverse=True)[:2]
report.append("\nTWO STRONGEST ABSOLUTE OFF-DIAGONAL CORRELATIONS")
for a, b, value, absolute in top_two:
    report.append(f"{a} vs {b}: correlation={value:.4f}, abs={absolute:.4f}")

# At least four multivariate charts, each accompanied by text in this report.
plt.figure(figsize=(7, 5))
sns.barplot(data=clean, x="sex", y="survived", hue="pclass", ci=None)
plt.ylabel("Survival rate")
plt.title("Survival by Sex and Passenger Class")
plt.tight_layout()
plt.savefig(OUTPUT / "story_1_survival_sex_class.png")
plt.close()
report.append(
    "\nChart 1 interpretation: Survival varies across sex and passenger class. "
    "The grouped bars allow the interaction between these two variables to be compared directly."
)

plt.figure(figsize=(7, 5))
sns.boxplot(data=clean, x="pclass", y="fare")
plt.title("Fare Distribution by Passenger Class")
plt.tight_layout()
plt.savefig(OUTPUT / "story_2_fare_class.png")
plt.close()
report.append(
    "\nChart 2 interpretation: Passenger class is associated with the distribution of fares. "
    "The box plots show both central tendency and spread within each class."
)

plt.figure(figsize=(7, 5))
sns.boxplot(data=clean, x="survived", y="age")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig(OUTPUT / "story_3_age_survival.png")
plt.close()
report.append(
    "\nChart 3 interpretation: The age distributions differ between survival outcomes. "
    "The box plots provide a compact comparison of the center and variability of age."
)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=clean,
    x="age",
    y="fare",
    hue="survived",
    style="pclass",
    alpha=0.65,
)
plt.title("Age, Fare, Class and Survival")
plt.tight_layout()
plt.savefig(OUTPUT / "story_4_age_fare_survival.png")
plt.close()
report.append(
    "\nChart 4 interpretation: Age and fare form a two-dimensional view of passenger characteristics. "
    "Adding survival and class visually shows how several variables interact rather than examining them independently."
)

# Exploratory standardization on the full cleaned DataFrame.
scaler = StandardScaler()
clean[["age_z", "fare_z"]] = scaler.fit_transform(clean[["age", "fare"]])

report.append("\nEDA STANDARDIZATION CHECK")
report.append(
    clean[["age_z", "fare_z"]].agg(["mean", "std"]).to_string()
)

# Save the cleaned data for inspection; modeling still reads the required CSV.
clean.to_csv(OUTPUT / "cleaned_titanic.csv", index=False)

(OUTPUT / "eda_report.txt").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print(f"\nEDA artifacts saved in {OUTPUT}")

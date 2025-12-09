import sys
import zlib
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

current_file = Path(__file__).resolve()
src_directory = current_file.parent.parent
sys.path.append(str(src_directory))

from income_predict.evaluation import evaluate_predictions
from income_predict.feature_engineering import SimpleStandardScaler

TARGET = "high_income"

parquet_path = src_directory / "data" / "cleaned_census_income.parquet"
df = pd.read_parquet(parquet_path)


def split_data_with_id_hash(data, test_ratio, id_column):
    def test_set_check(identifier):
        return (
            zlib.crc32(bytes(str(identifier), "utf-8")) & 0xFFFFFFFF
            < test_ratio * 2**32
        )

    ids = data[id_column]
    in_test_set = ids.apply(test_set_check)
    return data.loc[~in_test_set], data.loc[in_test_set]


train, test = split_data_with_id_hash(df, 0.2, "unique_id")

train_y = train[TARGET]
train_X = train.drop(columns=[TARGET, "unique_id"])

numeric_features = [
    "age",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

categorical_features = [
    "work_class",
    "education",
    # 'marital_status',
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", SimpleStandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# GLM Classifier Pipeline
glm_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", SGDClassifier(loss="log_loss", max_iter=1000)),
    ]
)

test_y = test[TARGET]
test_X = test.drop(columns=[TARGET, "unique_id"])

glm_pipeline.fit(train_X, train_y)
preds = glm_pipeline.predict(test_X)
acc = accuracy_score(test_y, preds)
baseline_clf = glm_pipeline.named_steps["classifier"]
baseline_params = {
    "classifier__alpha": baseline_clf.alpha,
    "classifier__l1_ratio": baseline_clf.l1_ratio,
}

print(f"GLM Baseline Accuracy: {acc:.4f}")
print(f"Baseline Params: {baseline_params}")

# Tuning GLM with Randomized Search
param_dist = {
    "classifier__l1_ratio": uniform(0, 1),
    "classifier__alpha": loguniform(1e-4, 1e-1),
}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    glm_pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=cv_strategy,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
)
random_search.fit(train_X, train_y)

print(f"GLM Tuned Accuracy: {random_search.best_score_:.4f}")
print(f"Tuned Params: {random_search.best_params_}")

# Evaluate Tuned GLM
test_eval_df = test_X.copy()
test_eval_df[TARGET] = test_y.values
test_eval_df["glm_preds"] = random_search.best_estimator_.predict_proba(test_X)[:, 1]
glm_eval = evaluate_predictions(test_eval_df, TARGET, preds_column="glm_preds")
print("\nTuned GLM Evaluation Metrics:")
print(glm_eval)


# LGBM Classifier Pipeline
lgbm_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(objective="binary", random_state=42, verbose=-1)),
    ]
)

lgbm_pipeline.fit(train_X, train_y)
preds = lgbm_pipeline.predict(test_X)
acc = accuracy_score(test_y, preds)
baseline_clf = lgbm_pipeline.named_steps["classifier"]
baseline_params = {
    "classifier__learning_rate": baseline_clf.learning_rate,
    "classifier__min_child_weight": baseline_clf.min_child_weight,
    "classifier__n_estimators": baseline_clf.n_estimators,
    "classifier__num_leaves": baseline_clf.num_leaves,
}

print(f"LGBM Baseline Accuracy: {acc:.4f}")
print(f"Baseline Params: {baseline_params}")

# Tuning LGBM with Randomized Search
param_dist = {
    "classifier__learning_rate": loguniform(0.01, 0.2),
    "classifier__n_estimators": randint(50, 200),
    "classifier__num_leaves": randint(10, 60),
    "classifier__min_child_weight": loguniform(0.0001, 0.002),
}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search_lgbm = RandomizedSearchCV(
    lgbm_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv_strategy,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
)
random_search_lgbm.fit(train_X, train_y)

print(f"LGBM Tuned Accuracy: {random_search_lgbm.best_score_:.4f}")
print(f"Tuned Params: {random_search_lgbm.best_params_}")

# Evaluate Tuned LGBM
test_eval_df["lgbm_preds"] = random_search_lgbm.best_estimator_.predict_proba(test_X)[
    :, 1
]
lgbm_eval = evaluate_predictions(test_eval_df, TARGET, preds_column="lgbm_preds")
print("\nTuned LGBM Evaluation Metrics:")
print(lgbm_eval)

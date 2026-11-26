
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_spend",
    "support_tickets",
]

CATEGORICAL_FEATURES = ["contract_type"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline():
    """Создаём ML-пайплайн с обработкой признаков."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="error")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    return model


def train_model(df, test_size=0.2, random_state=42):
    """Обучаем модель и возвращаем тестовую выборку."""

    X = df[FEATURES]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def predict_churn(model, df):

    return model.predict(df[FEATURES])


def predict_churn_probability(model, df):
    """Возвращаем вероятность оттока клиента"""

    return model.predict_proba(df[FEATURES])[:, 0]

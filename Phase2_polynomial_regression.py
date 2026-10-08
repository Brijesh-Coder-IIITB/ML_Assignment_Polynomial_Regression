# 1. import libraries and define parameters
from pathlib import Path
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV

ROLL = "BT2024263"
DEGREES = list(range(1, 13))
ALPHAS = np.logspace(-3, 3, 25)


# 2. locate the required dataset files
def find_file(filename: str) -> Path:
    candidates = []

    if len(sys.argv) > 1:
        candidates.append(
            Path(sys.argv[1]).expanduser().resolve() / filename
        )

    here = Path(__file__).resolve().parent
    candidates.extend([
        here / filename,
        Path.cwd() / filename
    ])

    for path in candidates:
        if path.is_file():
            return path

    searched = "\n".join(str(p) for p in candidates)

    raise FileNotFoundError(
        f"could not find {filename}. put the required dataset beside this script "
        f"or pass its folder as the first command-line argument.\nsearched:\n{searched}"
    )


# 3. load phase 2 data and prepare the regression model
def main() -> None:
    train_path = find_file(f"{ROLL}_train_var2.csv")
    test_path = find_file(f"{ROLL}_test_var2.csv")

    train = np.genfromtxt(train_path, delimiter=",", names=True)
    test = np.genfromtxt(test_path, delimiter=",", names=True)

    feature_names = [f"x{i}" for i in range(1, 4)]

    X = np.column_stack([
        train[name] for name in feature_names
    ])

    y = train["y"]

    X_test = np.column_stack([
        test[name] for name in feature_names
    ])

    model = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge())
    ])


    # 4. select the polynomial degree and regularization using cross-validation
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=model,
        param_grid={
            "poly__degree": DEGREES,
            "ridge__alpha": ALPHAS
        },
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X, y)


    # 5. generate and save the final phase 2 predictions
    prediction = search.best_estimator_.predict(X_test)

    output_path = (
        Path(__file__).resolve().parent
        / f"{ROLL}_pred_var2.csv"
    )

    np.savetxt(
        output_path,
        prediction.reshape(-1, 1),
        delimiter=",",
        header="y",
        comments="",
        fmt="%.15g",
    )

    print("Phase 2: Subterranean Thermal Reservoir Mapping")
    print(f"Training samples: {len(y)}")
    print(f"Test samples: {len(X_test)}")
    print(
        f"Best polynomial degree: "
        f"{search.best_params_['poly__degree']}"
    )
    print(
        f"Best Ridge alpha: "
        f"{search.best_params_['ridge__alpha']:.12g}"
    )
    print(
        f"Best 5-fold CV MSE: "
        f"{-search.best_score_:.12f}"
    )
    print(f"Prediction file: {output_path}")
    print("Output format: one column named y, 1000 predictions")


if __name__ == "__main__":
    main()
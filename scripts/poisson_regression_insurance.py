# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/poisson_regression_insurance.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="8dBgvvOgLktb"
# # Poisson regression for predicting insurance claim rates
#
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html?highlight=poisson%20regression
#

# + id="seiYZOhfLiZ8"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# + colab={"base_uri": "https://localhost:8080/", "height": 344} id="d16kYpGZQe3w" outputId="d3bf5c00-d8cc-443c-96e7-0c538b31d9e4"
# !pip install -U scikit-learn

# + colab={"base_uri": "https://localhost:8080/"} id="FNjDwPtqQYb6" outputId="d83e8e4c-dab9-4ede-eb38-4cf0d40c79c1"
import sklearn
print(sklearn.__version__)
from sklearn.linear_model import PoissonRegressor

# + [markdown] id="Di5Z5jWELxbz"
# # Data

# + colab={"base_uri": "https://localhost:8080/", "height": 419} id="E6FFyFEkLqeW" outputId="1fb39665-7789-4ec3-dd42-0ab005234b99"
from sklearn.datasets import fetch_openml


df = fetch_openml(data_id=41214, as_frame=True).frame
df

# + colab={"base_uri": "https://localhost:8080/", "height": 315} id="PQgENIDEMAZI" outputId="08e22267-b927-44f1-98a1-48d4239603f8"
df["Frequency"] = df["ClaimNb"] / df["Exposure"]

print("Average Frequency = {}"
      .format(np.average(df["Frequency"], weights=df["Exposure"])))

print("Fraction of exposure with zero claims = {0:.1%}"
      .format(df.loc[df["ClaimNb"] == 0, "Exposure"].sum() /
              df["Exposure"].sum()))

fig, ax = plt.subplots()
ax.set_title("Frequency (number of claims per year)")
_ = df["Frequency"].hist(bins=30, log=True, ax=ax)

# + colab={"base_uri": "https://localhost:8080/"} id="saTm9Z6KUxZO" outputId="83717f16-6204-474f-be10-99f9556dff6e"
print(df["Frequency"][:-20])

# + id="x7HJBPSOMWpX"

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.33, random_state=0)

# + colab={"base_uri": "https://localhost:8080/", "height": 265} id="_6TFmcx8TXNz" outputId="1053066f-e1aa-4f54-9fda-e4d9d610f158"
fig, ax = plt.subplots()
n_bins = 20
_ = df_test["Frequency"].hist(bins=np.linspace(-1, 30, n_bins), ax=ax)
ax.set_yscale('log')

# + [markdown] id="xJ2hTkNAMiun"
# # Feature engineering

# + id="qU8927WJMhwl"
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer


log_scale_transformer = make_pipeline(
    FunctionTransformer(np.log, validate=False),
    StandardScaler()
)

linear_model_preprocessor = ColumnTransformer(
    [
        ("passthrough_numeric", "passthrough",
            ["BonusMalus"]),
        ("binned_numeric", KBinsDiscretizer(n_bins=10),
            ["VehAge", "DrivAge"]),
        ("log_scaled_numeric", log_scale_transformer,
            ["Density"]),
        ("onehot_categorical", OneHotEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
    ],
    remainder="drop",
)

# + [markdown] id="TcLGQIaOMr9R"
# # Evaluation metrics

# + id="022GPhFtMuNx"
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance


def score_estimator(estimator, df_test):
    """Score an estimator on the test set."""
    y_pred = estimator.predict(df_test)

    print("MSE: %.3f" %
          mean_squared_error(df_test["Frequency"], y_pred,
                             sample_weight=df_test["Exposure"]))
    print("MAE: %.3f" %
          mean_absolute_error(df_test["Frequency"], y_pred,
                              sample_weight=df_test["Exposure"]))

    # Ignore non-positive predictions, as they are invalid for
    # the Poisson deviance.
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(f"WARNING: Estimator yields invalid, non-positive predictions "
              f" for {n_masked} samples out of {n_samples}. These predictions "
              f"are ignored when computing the Poisson deviance.")

    print("mean Poisson deviance: %.3f" %
          mean_poisson_deviance(df_test["Frequency"][mask],
                                y_pred[mask],
                                sample_weight=df_test["Exposure"][mask]))


# + id="5LvBbR3LM1yl"
def plot_predictions(models):
  nmodels = len(models)
  height = 5
  width = 5*(nmodels+1)
  fig, axes = plt.subplots(nrows=1, ncols=nmodels+1, figsize=(width,height), sharey=True)
  n_bins = 20
  df = df_test.copy()
  ax = axes[0]
  df["Frequency"].hist(bins=np.linspace(-1, 30, n_bins), ax=ax)
  ax.set_title("Data")
  ax.set_yscale('log')
  ax.set_xlabel("y (observed Frequency)")
  #ax.set_ylim([1e1, 5e5])
  ax.set_ylabel("#samples")

  for idx, model in enumerate(models):
    ax = axes[idx+1]
    y_pred = model.predict(df)
    pd.Series(y_pred).hist(bins=np.linspace(-1, 4, n_bins),ax=ax)
    ax.set(
        title=model[-1].__class__.__name__,
        yscale='log',
        xlabel="y_pred (predicted expected Frequency)"
    )

  plt.tight_layout()


# + [markdown] id="AaRi1NEnMZra"
# # Dummy model
#
# Just predicts overall mean.

# + id="qAsIMLW9Y_2z"
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

dummy = Pipeline([
    ("preprocessor", linear_model_preprocessor),
    ("regressor", DummyRegressor(strategy='mean')),
]).fit(df_train, df_train["Frequency"])

# + colab={"base_uri": "https://localhost:8080/"} id="Uc4Vv_1lZC09" outputId="2325c17e-c01b-4ec4-868b-f667bfcf8e31"
y_pred = dummy.predict(df_test)
print(y_pred[0:5])

# + [markdown] id="eco9nvvZai8B"
# We need to weight the examples by exposure, for reasons explained here:
# https://github.com/scikit-learn/scikit-learn/issues/18059

# + id="n55jU9YtZGHO"
# weighted version
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

dummy = Pipeline([
    ("preprocessor", linear_model_preprocessor),
    ("regressor", DummyRegressor(strategy='mean')),
]).fit(df_train, df_train["Frequency"],
       regressor__sample_weight=df_train["Exposure"])

# + colab={"base_uri": "https://localhost:8080/"} id="-MVrzS0wPIDA" outputId="eb21e9f9-e628-4fb4-fd27-15aac228cc7f"
y_pred = dummy.predict(df_test)
print(y_pred[0:5])

# + colab={"base_uri": "https://localhost:8080/"} id="lgcLhhZSNhMZ" outputId="6e67a6f9-1252-4490-fe42-3c83f5364946"
print("Constant mean frequency evaluation:")
score_estimator(dummy, df_test)

# + [markdown] id="Acsr7ZpFPhVP"
# # Linear regression
#
# Linear regression is an okay baseline, but not the most appropriate model for count data... We see the L2 regularization to a small value, since the training set is large.
#

# + id="HHILjZFHNp2k"
from sklearn.linear_model import Ridge


ridge_glm = Pipeline([
    ("preprocessor", linear_model_preprocessor),
    ("regressor", Ridge(alpha=1e-6)),
]).fit(df_train, df_train["Frequency"],
       regressor__sample_weight=df_train["Exposure"])

# + colab={"base_uri": "https://localhost:8080/"} id="3jq8dwgaPo2k" outputId="159e3412-9a86-493d-9436-cb74d598c274"
print("Ridge evaluation:")
score_estimator(ridge_glm, df_test)

# + [markdown] id="C92kplS5PuQh"
# # Poisson linear regression
#
# We set the L2 regularizer to the same value as in ridge regression, but divided by the number of training samples, since poisson regression penalizes the average log likelihood.
#

# + colab={"base_uri": "https://localhost:8080/"} id="HUCepZMsPwZP" outputId="627a856c-cbf0-412d-b639-d1702e471389"
from sklearn.linear_model import PoissonRegressor

n_samples = df_train.shape[0]

poisson_glm = Pipeline([
    ("preprocessor", linear_model_preprocessor),
    ("regressor", PoissonRegressor(alpha=1e-12, max_iter=300))
])
poisson_glm.fit(df_train, df_train["Frequency"],
                regressor__sample_weight=df_train["Exposure"])


# + colab={"base_uri": "https://localhost:8080/"} id="m0Lvr89wP9F7" outputId="38a0cb22-dcc1-4ea5-81e8-e6a0bfb76832"

print("PoissonRegressor evaluation:")
score_estimator(poisson_glm, df_test)

# + [markdown] id="Z2cMUYJ8RQoe"
# # Comparison

# + colab={"base_uri": "https://localhost:8080/", "height": 369} id="zW6qZ6gMRReW" outputId="c922f9e6-b257-47da-88d8-05904ffa8d23"
plot_predictions([ridge_glm, poisson_glm])

# + colab={"base_uri": "https://localhost:8080/", "height": 299} id="43Y-puKOSXOs" outputId="7192b9ff-fda7-4b1d-dce5-b669bf253fed"
plot_predictions([dummy, ridge_glm, poisson_glm])
plt.savefig('poisson_regr_insurance_pred.pdf')

# + colab={"base_uri": "https://localhost:8080/"} id="ZdS34iAQbL6D" outputId="0269d5d6-1543-41e0-8822-a1f7d68b33e4"
for model in [dummy, ridge_glm, poisson_glm]:
  print(model[-1].__class__.__name__)
  score_estimator(model, df_test)


# + [markdown] id="9M5eRU8ORcnZ"
# # Calibration plot

# + id="3-4QcAI_RVtZ"
from sklearn.utils import gen_even_slices


def _mean_frequency_by_risk_group(y_true, y_pred, sample_weight=None,
                                  n_bins=100):
    """Compare predictions and observations for bins ordered by y_pred.

    We order the samples by ``y_pred`` and split it in bins.
    In each bin the observed mean is compared with the predicted mean.

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred: array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,)
        Sample weights.
    n_bins: int
        Number of bins to use.

    Returns
    -------
    bin_centers: ndarray of shape (n_bins,)
        bin centers
    y_true_bin: ndarray of shape (n_bins,)
        average y_pred for each bin
    y_pred_bin: ndarray of shape (n_bins,)
        average y_pred for each bin
    """
    idx_sort = np.argsort(y_pred)
    bin_centers = np.arange(0, 1, 1/n_bins) + 0.5/n_bins
    y_pred_bin = np.zeros(n_bins)
    y_true_bin = np.zeros(n_bins)

    for n, sl in enumerate(gen_even_slices(len(y_true), n_bins)):
        weights = sample_weight[idx_sort][sl]
        y_pred_bin[n] = np.average(
            y_pred[idx_sort][sl], weights=weights
        )
        y_true_bin[n] = np.average(
            y_true[idx_sort][sl],
            weights=weights
        )
    return bin_centers, y_true_bin, y_pred_bin


def plot_calibration(models):
  print(f"Actual number of claims: {df_test['ClaimNb'].sum()}")
  nmodels = len(models)
  height = 5
  width = 5*(nmodels+1)
  fig, ax = plt.subplots(nrows=1, ncols=nmodels, figsize=(width, height))
  plt.subplots_adjust(wspace=0.3)
  for axi, model in zip(ax.ravel(), models):
      y_pred = model.predict(df_test)
      y_true = df_test["Frequency"].values
      exposure = df_test["Exposure"].values
      q, y_true_seg, y_pred_seg = _mean_frequency_by_risk_group(
          y_true, y_pred, sample_weight=exposure, n_bins=10)

      # Name of the model after the estimator used in the last step of the
      # pipeline.
      print(f"Predicted number of claims by {model[-1]}: "
            f"{np.sum(y_pred * exposure):.1f}")

      axi.plot(q, y_pred_seg, marker='x', linestyle="--", label="predictions")
      axi.plot(q, y_true_seg, marker='o', linestyle="--", label="observations")
      axi.set_xlim(0, 1.0)
      axi.set_ylim(0, 0.5)
      axi.set(
          title=model[-1],
          xlabel='Fraction of samples sorted by y_pred',
          ylabel='Mean Frequency (y_pred)'
      )
      axi.legend()
  plt.tight_layout()



# + colab={"base_uri": "https://localhost:8080/", "height": 367} id="Pkkv_nMSSFu_" outputId="9edec6ee-5b71-4bb5-e9aa-cd439dc1b792"
plot_calibration([dummy, ridge_glm, poisson_glm])
plt.savefig('poisson_regr_insurance.pdf')

# + id="n7oloi6SSH5W"


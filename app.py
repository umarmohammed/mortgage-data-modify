from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric


perf_metrics = {"Accuracy": metrics.accuracy_score,
                "Precision": metrics.precision_score,
                "Recall": metrics.recall_score,
                "F1-Score": metrics.f1_score,
                }


def xor(xs, ys):
    return (~xs & ys) | (xs & ~ys)


def subtractClampZero(x, y):
    return x - np.minimum(x, y)


@ignore_warnings(category=ConvergenceWarning)
def get_stats(df):

    def get_performance_metrics():
        return {x: perf_metrics[x](y.values.ravel(), ypred_class) for x in perf_metrics}

    def get_fairness_metrics():
        def get_bldm_metrics():

            metric_BLDM = BinaryLabelDatasetMetric(
                dataset, unprivileged_group, privileged_group)
            return {"Statistical Parity Difference": metric_BLDM.statistical_parity_difference(), "Disparate Impact": metric_BLDM.disparate_impact()}

        def get_cm_metrics():
            df_pred = X.copy()
            df_pred[df.columns[-1]] = np.expand_dims(ypred_class, axis=1)

            dataset_pred = BinaryLabelDataset(df=df_pred, label_names=[
                'action_taken_name'], protected_attribute_names=['applicant_sex_name_Female'])

            metric_CM = ClassificationMetric(
                dataset, dataset_pred, privileged_groups=privileged_group, unprivileged_groups=unprivileged_group)

            return {
                "Equal Opportunity Difference":   metric_CM.equal_opportunity_difference(),
                'Average Odds Difference': metric_CM.average_odds_difference(),
                "Accuracy Male": metric_CM.accuracy(privileged=True),
                "Accuracy Female":  metric_CM.accuracy(privileged=False)
            }

        dataset = BinaryLabelDataset(df=df, label_names=[
            'action_taken_name'], protected_attribute_names=['applicant_sex_name_Female'])

        privileged_group = [{'applicant_sex_name_Female': 0}]
        unprivileged_group = [{'applicant_sex_name_Female': 1}]

        return {**get_bldm_metrics(), **get_cm_metrics()}

    def get_misclassified_indexes():
        t = 0.65  # only consider miclassified above a threshold
        scores = np.abs(1*np.array(y) - np.array(ypred_prob))

        return [df.index[i]
                for i, x in enumerate(xor(ypred_class, y.values.ravel())) if x and scores[i] > t]

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    lr = LogisticRegression(
        random_state=10, solver="lbfgs", penalty="none")
    lr = lr.fit(X, y.values.ravel())

    ypred_class = lr.predict(X)
    ypred_prob = lr.predict_proba(X).ravel()[1::2]

    # bar = get_misclassified_indexes()

    return (
        get_performance_metrics(),
        get_fairness_metrics(),
        get_misclassified_indexes()
    )


def print_stats(df, performance, fairness, misclassifiedIndexes):
    print('-'*80)
    print(
        f"performance: {performance} | fairness: {fairness} | Total rows : {df.shape[0]} | Misclassified rows: {len(misclassifiedIndexes)}")
    print('-'*80)


def remove_misclassified(df, misclassifiedIndexes, malePercentage = None):

    # We don't want to remove too much data
    maxPercentageToRemove = 0.05
    maxNumberToRemove = maxPercentageToRemove * len(df)
    diff = subtractClampZero(len(misclassifiedIndexes),
                             maxNumberToRemove)
    misclassifiedIndexes = misclassifiedIndexes[int(diff):]

    def get_indexes(col):
        misclassifiedDf = df.loc[misclassifiedIndexes]

        return misclassifiedDf.loc[misclassifiedDf[col] == 1.0].index

    def adjust_misclassifiedIndexes(maleIndexes, femaleIndexes, targetMalePercentage):

        def getNumFemalesToRemove():
            targetTotal = actualNumMales / targetMalePercentage
            return 0 if targetNumMales < actualNumMales else len(misclassifiedIndexes) - targetTotal

        targetNumMales = len(misclassifiedIndexes) * targetMalePercentage
        actualNumMales = len(maleIndexes)

        nMalesToRemove = subtractClampZero(actualNumMales, targetNumMales)
        nFemalesToRemove = getNumFemalesToRemove()

        return list(maleIndexes[int(nMalesToRemove):]) + list(femaleIndexes[int(nFemalesToRemove):])

    maleIndexes = get_indexes("applicant_sex_name_Male")
    femaleIndexes = get_indexes("applicant_sex_name_Female")
    
    return df.drop(misclassifiedIndexes) if malePercentage is None else df.drop(adjust_misclassifiedIndexes(maleIndexes, femaleIndexes, malePercentage))


def modify_data(df):
    # stats = (df,) + get_stats(df)
    performance, fairness, misclassifiedIndexes = get_stats(df)
    print_stats(df, performance, fairness, misclassifiedIndexes)
    return df if performance["F1-Score"] > 0.63 else modify_data(remove_misclassified(df, misclassifiedIndexes))


# I originally pickled this with the X,y seperated. Not sure if this was a good idea.
X, y = load("mortgage_data_preprocess.pkl.gz")
# Probably best to concat them again as it makes the rest of the implementation less fiddly
dfOrig = pd.concat([X, y], axis=1)

dfMod = modify_data(dfOrig)
dump(dfMod, "mortgage_data_modified.pkl.gz")


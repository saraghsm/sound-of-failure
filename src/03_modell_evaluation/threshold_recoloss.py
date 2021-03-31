import numpy as np
##########################################################
# Import sklearn libraries
##########################################################
from sklearn.metrics import precision_recall_fscore_support
##########################################################
# Import pandas libraries
##########################################################
import pandas as pd


##########################################################
# Threshold definitions
##########################################################
def threshold_median_iqr(validation_loss, times_iqr):
    """
    Calculates loss threshold from median and interquartile
    range (IQR) of the validation reconstruction loss.
    Threshold = median + (times_iqr * IQR)
    Returns the predicted test labels after thresholding.

    validation_loss (array): reconstruction loss on the
                              validation data

    times_iqr (float): Number to multiply to the IQR for
                        setting the threshold.
    """
    loss_median = np.median(validation_loss)
    loss_iqr = np.subtract(*np.percentile(validation_loss,
                                          [75, 25]
                                          ))

    thres = loss_median + (times_iqr * loss_iqr)
    return thres


def threshold_percentile(validation_loss, percentile):
    """
    Calculates loss threshold based on q-th percentile
    of the validation reconstruction loss. Returns the
    predicted test labels after thresholding

    validation_loss (array): reconstruction loss on the
                              validation data
    percentile (int): Between 0 and 100. The percentile
                      above which the losses are outliers.
    """
    thres = np.percentile(validation_loss, percentile)
    return thres


##########################################################
# Predicted labels for a given threshold
##########################################################
def predicted_labels(reco_loss, thres):
    if (thres < reco_loss.min()) or (thres > reco_loss.max()):
        print(
            "Warning: The threshold is out of bounds from the "
            "range of the reconstruction loss. Are you sure your "
            "sample consists of only one class (either normal or abnormal)? "
            "If not, check the threshold calculation again...")
    return reco_loss > thres


##########################################################
# Precision, Recall and Accuracy
##########################################################
def calculate_metrics(true_test_labels, pred_test_labels):
    """
    Calculate the precision, recall and accuracy
    given the true and predicted test labels.

    NOTE: The predicted labels should be 0 or 1, i.e
    thresholding on the reconstruction loss should be
    applied before calculating these metrics.
    """
    weighted_f1 = precision_recall_fscore_support(true_test_labels,
                                                  pred_test_labels,
                                                  beta=3,
                                                  average='binary')

    precision, recall, fbeta_score, _ = weighted_f1

    return precision, recall, fbeta_score


##########################################################
# Metrics for a vairable threshold
##########################################################
def metrics_by_variable_threshold(validation_loss,
                                  test_loss,
                                  test_labels,
                                  measure='percentile',
                                  criterion_range=None,
                                  ):
    """
    Return a pd.DataFrame containing precision, recall,
    accuracy for a variable threshold. The 'measure'
    indicates what method is being used for thresholding:
    percentile or std. dev from the mean

    :param validation_loss: Validation reco loss
    :param test_loss: Test reco loss
    :param test_labels: True test labels (binary)
    :param measure: 'percentile' or 'stddev'
    :param criterion_range: Range of percentiles or std.dev
                            to try for thresholding.

    :return: pd.DataFrame containing 4 columns -
        - values of the possible thresholds
        - Precision scores
        - Recall scores
        - Accuracy scores
    """
    threshold = []
    precision = []
    recall = []
    fbeta_score = []

    for criterion in criterion_range:
        if measure == 'percentile':
            thres = threshold_percentile(validation_loss, criterion)
            pred_test_labels = predicted_labels(test_loss, thres)
        elif measure == 'iqr':
            thres = threshold_median_iqr(validation_loss, criterion)
            pred_test_labels = predicted_labels(test_loss, thres)

        else:
            print("measure should be either 'percentile' or 'iqr'. Exiting...")
            break

        prec, rec, f_scr = calculate_metrics(test_labels, pred_test_labels)

        threshold.append(thres)
        precision.append(prec)
        recall.append(rec)
        fbeta_score.append(f_scr)
        # accuracy.append(acc)

    metrics_df = pd.DataFrame({'' + measure: criterion_range,
                               'Threshold': threshold,
                               'Precision': precision,
                               'Recall': recall,
                               'Fbeta_score': fbeta_score})

    return metrics_df


def get_threshold_criterion(metric_df, metric_to_optimize):
    """
    Return the criterion for thresholding (how many percentile
    or how much fraction of the interquartile range) that
    maximizes a metric given by 'metric_to_optimize'. Currently,
    there are 3 options : 'Precision', 'Recall', 'Fbeta_score'

    :param metric_df: A Dataframe containing the metrics for a
                      variable threshold (return from the function
                      'metrics_by_variable_threshold')

    :param metric_to_optimize: Name of the metric to maximize
                                'Precision', 'Recall', 'Fbeta_score'
                                Recommended: 'Fbeta_score'

    :return: A criterion for thresholding
    """
    # The first index that has the maximum value of the 'metric_to_optimize'
    idx_with_max_metric = metric_df.loc[:, metric_to_optimize].idxmax()
    max_metric = metric_df.loc[:, metric_to_optimize].max()
    criterion_name = metric_df.columns[0]

    criterion = np.round(
        metric_df.loc[idx_with_max_metric, criterion_name], 2)

    print("The threshold that provides the best {} with value "
          "{} corresponds to {} {}".format(metric_to_optimize,
                                           max_metric,
                                           criterion,
                                           criterion_name))
    return {criterion_name: criterion}


def set_threshold(validation_loss, criterion_dict):
    """
    Set a threshold based on a given measure and criterion.
    'measure' can be 'iqr' or 'percentile'.
    'criterion' is the value of 'iqr' or 'percentile'.
    """
    criterion = list(criterion_dict.keys())[0]

    if criterion == 'iqr':
        thres = threshold_median_iqr(validation_loss, criterion_dict[criterion])
    elif criterion == 'percentile':
        thres = threshold_percentile(validation_loss, criterion_dict[criterion])
    return thres
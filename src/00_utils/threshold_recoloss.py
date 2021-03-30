##########################################################
# Import sklearn libraries
##########################################################
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
##########################################################
# Import pandas libraries
##########################################################
import pandas as pd

##########################################################
# Threshold definitions
##########################################################
def threshold_stddev(validation_loss, no_of_std):
    """
    Calculates loss threshold from mean and std.
    deviation of the validation reconstruction loss.
    Returns the predicted test labels after thresholding.

    validation_loss (array): reconstruction loss on the
                              validation data
    test_loss (array): reconstruction loss on the
                              test data

    no_of_std (float): Number of std dev away from mean
                        that should be used as threshold.
    """
    loss_mean = np.median(validation_loss)
    loss_stddev = np.std(validation_loss)

    thres = loss_mean + (no_of_std * loss_stddev)
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
    precision = precision_score(true_test_labels,
                                pred_test_labels)

    recall = recall_score(true_test_labels,
                          pred_test_labels)

    accuracy = accuracy_score(true_test_labels,
                              pred_test_labels)

    return precision, recall, accuracy


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
    accuracy = []

    for criterion in criterion_range:
        if measure == 'percentile':
            thres = threshold_percentile(validation_loss, criterion)
            pred_test_labels = test_loss > thres
        elif measure == 'stddev':
            thres = threshold_stddev(validation_loss, criterion)
            pred_test_labels = test_loss > thres

        else:
            print("measure should be either 'percentile' or 'stdddev'. Exiting...")
            break

        prec, rec, acc = calculate_metrics(test_labels, pred_test_labels)

        threshold.append(thres)
        precision.append(prec)
        recall.append(rec)
        accuracy.append(acc)

    metrics_df = pd.DataFrame({'Threshold_' + measure: threshold,
                               'Precision': precision,
                               'Recall': recall,
                               'Accuracy': accuracy})

    return metrics_df


def optimal_threshold(metric, target_metric_val, metric_df):
    """
    When test data is available, the optimal threshold is
    basically the value of the reco loss that best separates
    test normal and test anomalous data.

    This function offers two choice of metrics: accuracy and
    precision-recall.
    The aim is to achieve EITHER a high test prediction accuracy
    OR a high recall and precision score.

    :param metric (str): 'prec_recall' or 'accuracy'
    :param target_metric_val (list): [target_precision, target_recall]
                                     OR [target_accuracy]
    :param metric_df: A Dataframe containing the metrics for a
                      variable threshold (return from the function
                      'metrics_by_variable_threshold')

    :return: A DataFrame containing all the possible
              optimal thresholds and a single optimal threshold
              calculated as the mean of all possible optimal thresholds.
    """
    if metric == 'prec_recall':
        try:
            assert len(target_metric_val) == 2
            target_precision, target_recall = target_metric_val
            mask = (metric_df.Precision >= target_precision) & (
                metric_df.Recall >= target_recall)
            optim_thres_df = metric_df.loc[mask]
            optim_thres = optim_thres_df.iloc[:, 0].mean()
            return optim_thres_df, optim_thres

        except AssertionError:
            print(
                "'target_metric_val' should be of length 2 including precision and recall")

    elif metric == 'accuracy':
        try:
            assert len(target_metric_val) == 1
            target_accuracy = target_metric_val[0]
            mask = round(metric_df.Accuracy, 2) >= target_accuracy
            optim_thres_df = metric_df.loc[mask]
            optim_thres = optim_thres_df.iloc[:, 0].mean()
            return optim_thres_df, optim_thres

        except AssertionError:
            print(
                "'target_metric_val' should be a list of length 1 containing target accuracy")

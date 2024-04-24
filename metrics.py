import krippendorff
from lifelines.utils import concordance_index as ci
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score
import pickle

def acc(preds, ground_truths):
    return accuracy_score(ground_truths, preds)

def _prepare_data(preds, ground_truths, scalers):
    try:
        predictions = preds.cpu().numpy(force=True)
        ground_truths = ground_truths.cpu().numpy(force=True)
        scalers = scalers.cpu().numpy(force=True)
    except: # previous torch versions do not have force parameter
        predictions = preds.cpu().numpy()
        ground_truths = ground_truths.cpu().numpy()
        scalers = scalers.cpu().numpy()
    predictions = np.around(np.multiply(predictions,scalers)).astype(int)
    ground_truths = np.multiply(ground_truths,scalers).astype(int)
    return predictions, ground_truths

def _kappa_kohen(preds, ground_truths):
    return cohen_kappa_score(preds, ground_truths, weights="quadratic")
    
def spearman(preds, ground_truths, scalers):
    predictions, ground_truths = _prepare_data(preds, ground_truths, scalers)
    return spearmanr(predictions, ground_truths)[0]

def _kendall_rank(preds, ground_truths):
    return kendalltau(preds, ground_truths)[0]

def _krippendorff_alpha(preds, ground_truths):
    return krippendorff.alpha([preds, ground_truths], level_of_measurement="interval", value_domain=range(41))

def _concordance_index(preds, ground_truths):
    return ci(ground_truths, preds)

def _mean_squared_error(preds, ground_truths):
    return mean_squared_error(ground_truths, preds)

def get_eval_metrics(preds, ground_truths_p, scalers):
    predictions, ground_truths = _prepare_data(preds, ground_truths_p, scalers)
    return {"kappa_kohen": _kappa_kohen(predictions, ground_truths),
            "spearman": spearman(preds, ground_truths_p, scalers),
            "kendall_ranking_coefficient": _kendall_rank(predictions, ground_truths),
            "krippendorff_alpha": _krippendorff_alpha(predictions, ground_truths),
            "concordance_index": _concordance_index(predictions, ground_truths),
            "MSE": _mean_squared_error(predictions, ground_truths)}

"""The proposal's metrics, per test generator. AI-generated is the positive class."""
from sklearn.metrics import average_precision_score, roc_auc_score

from ror.data import FAKE, REAL


def score(labels, probs):
    pred = probs > 0.5
    return {
        "acc": (pred == labels).mean() * 100,
        "real_acc": (pred[labels == REAL] == REAL).mean() * 100,
        "fake_acc": (pred[labels == FAKE] == FAKE).mean() * 100,
        "ap": average_precision_score(labels, probs) * 100,
        "auc": roc_auc_score(labels, probs) * 100,
    }

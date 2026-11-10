"""Is each update's difference from B real? And is U1's accuracy problem the 0.5 cutoff?

    python -m ror.analysis

Paired comparisons: every model is compared with B on exactly the same test images (the damaged
copies on the 1,000 + 1,000 subset that every model was scored on). Each model's prediction for an
image is its P(AI) averaged over the 3 seeds, so the test is about the method, not one lucky seed.

- McNemar's test (pooled over the generators): of the images exactly one model gets right, does
  one model win significantly more of them?
- Bootstrap 95% confidence intervals for the differences in accuracy and AUC, mean over generators,
  resampling images within each generator (1,000 resamples).

Threshold analysis, on clean images: accuracy at the 0.5 cutoff next to the best possible cutoff
for each generator, and the median P(AI) of real and AI images. The best cutoff is chosen on the
test images, so it is a diagnostic only, never a result to report as accuracy.

Writes results/analysis_significance.csv and results/analysis_threshold.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2

from ror.collect import UNSEEN, models, predictions
from ror.config import GENERATORS, RESULTS
from ror.views import DEGRADED_TEST_VIEWS

PROTOCOLS = ["resize"] + DEGRADED_TEST_VIEWS
N_BOOTSTRAP = 1000
REFERENCE = "B"


def auc(labels, scores):
    """ROC-AUC from ranks (Mann-Whitney), fast enough to bootstrap. Ties get average ranks."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    _, first, counts = np.unique(scores[order], return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2 + 1
    ranks[order] = np.repeat(avg, counts)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mcnemar(correct_a, correct_b):
    """Chi-square McNemar test with continuity correction. Returns (only A right, only B right, p)."""
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    if a_only + b_only == 0:
        return a_only, b_only, 1.0
    stat = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
    return a_only, b_only, float(chi2.sf(stat, 1))


def _paired(reference, other):
    """Restrict two models' predictions to the images both were tested on: {gen: (labels, p_ref, p_other)}."""
    out = {}
    for generator in GENERATORS:
        rows_r, labels_r, probs_r = reference[generator]
        rows_o, labels_o, probs_o = other[generator]
        common, idx_r, idx_o = np.intersect1d(rows_r, rows_o, return_indices=True)
        assert (labels_r[idx_r] == labels_o[idx_o]).all()
        out[generator] = (labels_r[idx_r], probs_r[:, idx_r].mean(0), probs_o[:, idx_o].mean(0))
    return out


def compare(reference, other, rng):
    paired = _paired(reference, other)
    rows = {}
    for subset, generators in [("all 8", GENERATORS), ("7 unseen", UNSEEN)]:
        labels = np.concatenate([paired[g][0] for g in generators])
        correct_r = np.concatenate([(paired[g][1] > 0.5) == paired[g][0] for g in generators])
        correct_o = np.concatenate([(paired[g][2] > 0.5) == paired[g][0] for g in generators])
        ref_only, other_only, p = mcnemar(correct_r, correct_o)
        rows[subset] = {"generators": subset, "images": len(labels), "ref_only_right": ref_only,
                        "other_only_right": other_only, "mcnemar_p": p}

    # Bootstrap: resample images within each generator, average the per-generator differences.
    diffs = {g: [] for g in GENERATORS}
    point = {}
    for generator in GENERATORS:
        y, p_r, p_o = paired[generator]
        point[generator] = (((p_o > 0.5) == y).mean() - ((p_r > 0.5) == y).mean(), auc(y, p_o) - auc(y, p_r),
                            ((p_r > 0.5) == y).mean(), ((p_o > 0.5) == y).mean(), auc(y, p_r), auc(y, p_o))
        for _ in range(N_BOOTSTRAP):
            i = rng.integers(0, len(y), len(y))
            yi = y[i]
            if yi.min() == yi.max():
                continue
            diffs[generator].append((((p_o[i] > 0.5) == yi).mean() - ((p_r[i] > 0.5) == yi).mean(),
                                     auc(yi, p_o[i]) - auc(yi, p_r[i])))
    for subset, generators in [("all 8", GENERATORS), ("7 unseen", UNSEEN)]:
        n = min(len(diffs[g]) for g in generators)
        boot = np.mean([np.array(diffs[g][:n]) for g in generators], axis=0)  # [n, 2]
        mean_point = np.mean([point[g] for g in generators], axis=0) * 100
        lo, hi = np.percentile(boot * 100, [2.5, 97.5], axis=0)
        rows[subset] |= {
            "acc_ref": mean_point[2], "acc_other": mean_point[3], "acc_diff": mean_point[0],
            "acc_diff_lo": lo[0], "acc_diff_hi": hi[0],
            "auc_ref": mean_point[4], "auc_other": mean_point[5], "auc_diff": mean_point[1],
            "auc_diff_lo": lo[1], "auc_diff_hi": hi[1],
        }
    return list(rows.values())


def significance():
    rng = np.random.default_rng(0)
    rows = []
    for protocol in PROTOCOLS:
        reference = predictions(REFERENCE, protocol)
        for label in models():
            if label == REFERENCE:
                continue
            for row in compare(reference, predictions(label, protocol), rng):
                rows.append({"model": label, "protocol": protocol} | row)
            print(f"{protocol:7s} {label}: done")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "analysis_significance.csv", index=False)
    return table


def threshold():
    rows = []
    for label in models():
        for generator, (_, labels, probs) in predictions(label, "resize").items():
            p = probs.mean(0)
            cutoffs = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
            balanced = [((p[labels == 1] > c).mean() + (p[labels == 0] <= c).mean()) / 2 for c in cutoffs]
            best = int(np.argmax(balanced))
            rows.append({
                "model": label, "generator": generator,
                "acc_at_0.5": ((p > 0.5) == labels).mean() * 100,
                "best_cutoff": cutoffs[best], "acc_at_best_cutoff": balanced[best] * 100,
                "median_p_real": np.median(p[labels == 0]), "median_p_ai": np.median(p[labels == 1]),
            })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "analysis_threshold.csv", index=False)
    return table


def main():
    pd.set_option("display.width", 220)
    sig = significance()
    show = ["model", "protocol", "generators", "acc_diff", "acc_diff_lo", "acc_diff_hi", "auc_diff", "auc_diff_lo",
            "auc_diff_hi", "mcnemar_p"]
    print(f"\nEach model minus {REFERENCE}, seed-averaged predictions, 95% bootstrap CIs (percentage points):")
    print(sig[show].round(3).to_string(index=False))

    thr = threshold()
    print("\nThreshold diagnostic, clean images (the best cutoff is chosen on test data: diagnostic only):")
    print(thr.pivot(index="generator", columns="model", values="acc_at_0.5").round(1).to_string())
    print("\naccuracy at the best cutoff:")
    print(thr.pivot(index="generator", columns="model", values="acc_at_best_cutoff").round(1).to_string())
    print("\nmedian P(AI) of AI images (real images' median in brackets):")
    med = thr.assign(cell=thr.apply(lambda r: f"{r['median_p_ai']:.2f} ({r['median_p_real']:.2f})", axis=1))
    print(med.pivot(index="generator", columns="model", values="cell").to_string())


if __name__ == "__main__":
    main()

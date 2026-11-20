"""Build the combined development comparison from saved fold-level files.

Reads only saved fold_metrics.csv / per_class_metrics.csv files.
It does not train, refit, or score any model and does not load test rows.
Writes results/shared/combined_comparison/.
"""
import csv
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "shared" / "combined_comparison"
SIZES = [1486, 1481, 1481, 1470, 1463]
K = 5

P1, P2, P4 = R / "person1", R / "person2", R / "person4" / "pca_vs_original"
RF = R / "person3" / "random_forest"
CEN = R / "person3" / "centroid_sensitivity" / "fold_metrics.csv"

# label -> (owner, section, folder or (file, model name))
CONFIG_A = [
    ("Random forest (tuned)", "P3", "main", ("P3rf", "Tuned forest")),
    ("RBF SVM", "P2", "main", P2 / "rbf_svm_tuned"),
    ("Linear SVM", "P2", "main", P2 / "linear_svm_tuned"),
    ("Logistic regression", "P1", "main", P1 / "e1_lr_l2_3class"),
    ("Pruned DT", "P3", "main", ("P3rf", "Pruned tree")),
    ("KNN", "P2", "main", P2 / "knn_tuned"),
    ("Naive Bayes", "P1", "main", P1 / "e2_nb_b_signed_log_3class"),
    ("Majority floor", "P1", "floor", P1 / "e0_dummy_most_frequent_3class"),
    ("PCA probe: original 76", "P4", "PCA information probe", P4 / "original_76"),
    ("PCA probe: PCA 25", "P4", "PCA information probe", P4 / "pca_25"),
]
SAME = "same as A"
RETUNED_LINEAR = "retuned (no fixed-params run exists)"
# label -> (source, B settings). Configuration B reuses the A settings for every
# model except Linear SVM, which has no fixed-params run.
CONFIG_B = [
    ("Logistic regression", P1 / "e3_f56_no_centroid_lr_3class", SAME),
    ("Naive Bayes", P1 / "e3_f56_no_centroid_nb_3class", SAME),
    ("KNN", P2 / "knn_tuned_no_centroid_fixed_params", SAME),
    ("Linear SVM", P2 / "linear_svm_tuned_no_centroid", RETUNED_LINEAR),
    ("RBF SVM", P2 / "rbf_svm_tuned_no_centroid_fixed_params", SAME),
    ("Pruned DT", ("P3cen", "Pruned tree"), SAME),
    ("Random forest (tuned)", ("P3cen", "Tuned forest"), SAME),
]
# Person 2 reference: retuned B next to fixed-settings B.
P2_RETUNED_VS_FIXED = [
    ("KNN", P2 / "knn_tuned_no_centroid", P2 / "knn_tuned_no_centroid_fixed_params"),
    ("RBF SVM", P2 / "rbf_svm_tuned_no_centroid", P2 / "rbf_svm_tuned_no_centroid_fixed_params"),
]
METRICS = ["validation_macro_f1", "validation_accuracy",
           "validation_macro_precision", "validation_macro_recall",
           "train_macro_f1"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def folds(src, cfg):
    """Return the 5 fold rows (sorted by fold) for a source."""
    if isinstance(src, tuple):
        kind, name = src
        if kind == "P3rf":
            rows = [r for r in read(RF / "comparison_folds.csv") if r["model"] == name]
        else:
            rows = [r for r in read(CEN) if r["model"] == name
                    and r["configuration"] == cfg and r["diagnostic"] == "False"]
    else:
        rows = read(src / "fold_metrics.csv")
    rows.sort(key=lambda r: int(r["fold"]))
    ids = [int(r["fold"]) for r in rows]
    sizes = [int(r["n_validation"]) for r in rows]
    if ids != list(range(K)) or sizes != SIZES:
        sys.exit(f"CHECK FAILED for {src} {cfg}: folds {ids}, sizes {sizes}")
    return rows


def col(rows, c):
    return [float(r[c]) for r in rows]


def msd(v):
    return st.mean(v), st.stdev(v)


def t4_two_sided_p(t):
    """Two-sided p for Student t with 4 df (closed form of the CDF)."""
    t = abs(t)
    x = t / math.sqrt(4 + t * t)
    return 1 - x * (1 + (1 - x * x) / 2)


def nb_test(d, ratios):
    n = len(d)
    var = st.variance(d)
    r = st.mean(ratios)
    if var == 0:
        return float("nan"), float("nan")
    t = st.mean(d) / math.sqrt((1 / n + r) * var)
    return t, t4_two_sided_p(t)


def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def f(x, n=4):
    return f"{x:.{n}f}"


def main():
    # sanity check of the p-value formula: t=2.776445 -> p=0.05 at 4 df
    assert abs(t4_two_sided_p(2.776445) - 0.05) < 1e-4

    data, fm = {}, {}
    for label, owner, sec, src in CONFIG_A:
        fm[label] = folds(src, "A")
    for label, src, _ in CONFIG_B:
        if isinstance(src, tuple) or (src / "fold_metrics.csv").exists():
            data[label] = folds(src, "B")
        else:
            data[label] = None
    print("check ok: all inputs have folds 0-4 with sizes", SIZES)

    # 1. summary
    rows = []
    for label, owner, sec, src in CONFIG_A:
        r = fm[label]
        out = [label, owner, sec]
        for m in METRICS:
            mu, sd = msd(col(r, m))
            out += [f(mu), f(sd)]
        gaps = [a - b for a, b in zip(col(r, "train_macro_f1"), col(r, "validation_macro_f1"))]
        mu, sd = msd(gaps)
        out += [f(mu), f(sd)]
        rows.append(out)
    hdr = ["model", "owner", "section"]
    for m in METRICS + ["gap"]:
        hdr += [m + "_mean", m + "_sd"]
    write("summary_config_a.csv", hdr, rows)

    # 2. per class recall
    pc = []
    for label, owner, sec, src in CONFIG_A:
        if isinstance(src, tuple):
            rr = [x for x in read(RF / "per_class_metrics.csv") if x["model"] == src[1]]
        else:
            rr = read(src / "per_class_metrics.csv")
        for x in rr:
            pc.append([label, x["class"], f(float(x["recall"])), x["support"]])
    write("per_class_recall.csv", ["model", "class", "recall", "support"], pc)

    # 3. paired folds
    pairs = [("Random forest (tuned)", "RBF SVM"), ("Random forest (tuned)", "Logistic regression"),
             ("RBF SVM", "Logistic regression"), ("Pruned DT", "Logistic regression")]
    ratios = [(n_v / int(n_t)) for n_v, n_t in
              zip(SIZES, [r["n_train"] for r in fm["Logistic regression"]])]
    pr = []
    for a, b in pairs:
        for m in ["validation_macro_f1", "validation_accuracy"]:
            d = [x - y for x, y in zip(col(fm[a], m), col(fm[b], m))]
            t, p = nb_test(d, ratios)
            pr.append([a, b, m] + [f(x, 5) for x in d] +
                      [f(st.mean(d), 5), f(min(d), 5), f(max(d), 5),
                       sum(x > 0 for x in d), f(t, 3), f(p, 4)])
    write("paired_folds.csv",
          ["model_a", "model_b", "metric"] + [f"diff_fold{i}" for i in range(K)] +
          ["mean_diff", "min_diff", "max_diff", "folds_a_higher", "nb_corrected_t", "nb_two_sided_p"], pr)

    # 4. centroid
    cr = []
    for label, src, b_settings in CONFIG_B:
        a = col(fm[label], "validation_macro_f1") if label in fm else None
        if a is None:
            a = col(folds([c[3] for c in CONFIG_A if c[0] == label][0], "A"), "validation_macro_f1")
        if data[label] is None:
            cr.append([label, b_settings, f(st.mean(a)), "not run", "not run", "not run"])
            continue
        b = col(data[label], "validation_macro_f1")
        cr.append([label, b_settings, f(st.mean(a)), f(st.mean(b)), f(st.mean(b) - st.mean(a), 4),
                   sum(y < x for x, y in zip(a, b))])
    write("centroid_a_vs_b.csv",
          ["model", "B_settings", "macro_f1_A", "macro_f1_B", "diff_B_minus_A", "folds_B_lower"], cr)

    # 4b. Person 2 retuned vs fixed, for reference
    rf_rows = []
    for label, retuned, fixed in P2_RETUNED_VS_FIXED:
        a = col(fm[label], "validation_macro_f1")
        rt = col(folds(retuned, "B"), "validation_macro_f1")
        fx = col(folds(fixed, "B"), "validation_macro_f1")
        rf_rows.append([label, f(st.mean(a)), f(st.mean(rt)), f(st.mean(fx)),
                        f(st.mean(fx) - st.mean(rt), 4)])
    write("centroid_p2_retuned_vs_fixed.csv",
          ["model", "macro_f1_A", "macro_f1_B_retuned", "macro_f1_B_fixed_A_settings",
           "diff_fixed_minus_retuned"], rf_rows)
    for label, e in {"KNN": 0.6301, "RBF SVM": 0.7296}.items():
        got = [float(r[3]) for r in cr if r[0] == label][0]
        if abs(got - e) > 0.0005:
            print(f"B MISMATCH {label}: expected about {e}, got {got:.4f}")

    # expected values
    exp = {"Random forest (tuned)": .8156, "RBF SVM": .7888, "Linear SVM": .7750,
           "Logistic regression": .7721, "Pruned DT": .7716, "KNN": .7364,
           "Naive Bayes": .6699, "Majority floor": .2218}
    for label, e in exp.items():
        got = st.mean(col(fm[label], "validation_macro_f1"))
        if abs(got - e) > 0.0005:
            print(f"MISMATCH {label}: expected {e}, got {got:.4f}")
    d = [x - y for x, y in zip(col(fm["Random forest (tuned)"], "validation_macro_f1"),
                               col(fm["RBF SVM"], "validation_macro_f1"))]
    t, p = nb_test(d, ratios)
    print(f"RF-RBF mean {st.mean(d):.4f} (exp 0.0268), higher {sum(x > 0 for x in d)}/5, p {p:.4f} (exp ~0.009)")


if __name__ == "__main__":
    main()

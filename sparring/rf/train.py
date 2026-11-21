#!/usr/bin/env python3
"""Train a random forest on collect.py rows, score it on held-out opponent types, export JSON.

    python3 sparring/rf/train.py /tmp/rf/rows.pkl --export my_bots/tag_rf/data/rf.json

Offline question: does the forest predict P(opponent continues vs a bet) better than tag's own
running-frequency estimate, on opponent *types never used in training*?
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rf_features import RF_FEATURES  # noqa: E402


def load(path):
    rows = pickle.load(open(path, "rb"))
    X = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows])
    name = np.array([r[2] for r in rows])
    split = np.array([r[3] for r in rows])
    tag_p = np.array([r[5] for r in rows])
    tag_rs = np.array([r[6] for r in rows])
    return X, y, name, split, tag_p, tag_rs


def metrics(p, cont):
    p = np.clip(p, 0.02, 0.98)
    ll = -np.mean(cont * np.log(p) + (1 - cont) * np.log(1 - p))
    return ll, np.mean((p - cont) ** 2)


def cont_prob(model, X):
    return 1 - model.predict_proba(X)[:, list(model.classes_).index(0)]


def fit(X, y, **kw):
    params = dict(n_estimators=150, min_samples_leaf=25, max_depth=10, max_features=0.5, n_jobs=4,
                  random_state=0, class_weight=None)
    params.update(kw)
    return RandomForestClassifier(**params).fit(X, y)


def export(model, path, classes=(0, 1, 2)):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        val = t.value[:, 0, :]
        val = val / val.sum(axis=1, keepdims=True)
        full = np.zeros((len(val), len(classes)))
        for j, c in enumerate(model.classes_):
            full[:, list(classes).index(c)] = val[:, j]
        trees.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(),
                      "f": t.feature.tolist(), "t": [round(float(x), 4) for x in t.threshold],
                      "v": [[round(float(x), 3) for x in row] for row in full]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"features": list(RF_FEATURES), "trees": trees}, separators=(",", ":")))
    print(f"exported {len(trees)} trees, {Path(path).stat().st_size / 1e6:.1f} MB -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--export")
    ap.add_argument("--cv", action="store_true", help="group-CV a few hyperparameters on the train split")
    args = ap.parse_args()
    X, y, name, split, tag_p, tag_rs = load(args.rows)
    tr, te = split == "train", split == "test"
    print(f"train rows {tr.sum()} ({len(set(name[tr]))} opponents), test rows {te.sum()} ({sorted(set(name[te]))})")
    print("class mix train", np.bincount(y[tr]) / tr.sum(), " test", np.bincount(y[te], minlength=3) / te.sum())

    if args.cv:
        for leaf in (10, 25, 60):
            for depth in (6, 10, 16):
                lls = []
                for a, b in GroupKFold(5).split(X[tr], y[tr], name[tr]):
                    m = fit(X[tr][a], y[tr][a], min_samples_leaf=leaf, max_depth=depth)
                    lls.append(metrics(cont_prob(m, X[tr][b]), (y[tr][b] > 0).astype(float))[0])
                print(f"  leaf {leaf:>3} depth {depth:>2}: group-CV logloss {np.mean(lls):.4f}")

    model = fit(X[tr], y[tr])
    cont = (y[te] > 0).astype(float)
    p_rf = cont_prob(model, X[te])
    results = {"tag (current)": tag_p[te], "forest": p_rf, "blend 50/50": 0.5 * p_rf + 0.5 * tag_p[te],
               "constant (train mean)": np.full(te.sum(), (y[tr] > 0).mean())}
    print(f"\nHELD-OUT opponent types: P(continue) quality (lower is better)")
    print(f"{'estimator':<24}{'logloss':>9}{'brier':>8}")
    for k, p in results.items():
        ll, br = metrics(p, cont)
        print(f"{k:<24}{ll:>9.4f}{br:>8.4f}")
    n_faced = X[te][:, RF_FEATURES.index("n_faced")]
    print("\nby how many bets the opponent had faced so far (logloss tag / forest / blend):")
    for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 10 ** 6)):
        m = (n_faced >= lo) & (n_faced < hi)
        if m.sum() < 30:
            continue
        print(f"  faced {lo:>2}-{hi if hi < 10**6 else '+':<3} n={m.sum():>5}  "
              + "  ".join(f"{metrics(p[m], cont[m])[0]:.4f}" for p in (tag_p[te], p_rf, 0.5 * p_rf + 0.5 * tag_p[te])))
    print("\nper held-out opponent (logloss tag / forest):")
    for nm in sorted(set(name[te])):
        m = name[te] == nm
        print(f"  {nm:<14} n={m.sum():>5}  {metrics(tag_p[te][m], cont[m])[0]:.4f}  {metrics(p_rf[m], cont[m])[0]:.4f}")
    imp = sorted(zip(model.feature_importances_, RF_FEATURES), reverse=True)[:8]
    print("\ntop features:", ", ".join(f"{n} {v:.2f}" for v, n in imp))
    # raise share among continues
    c = (y[te] > 0)
    if c.sum() > 50:
        pr = model.predict_proba(X[te][c])
        rs = pr[:, list(model.classes_).index(2)] / np.maximum(1e-6, 1 - pr[:, list(model.classes_).index(0)])
        print(f"raise-share among continues: actual {np.mean(y[te][c] == 2):.3f}  tag {tag_rs[te][c].mean():.3f}  "
              f"forest {rs.mean():.3f}   brier tag {np.mean((tag_rs[te][c] - (y[te][c] == 2)) ** 2):.4f} "
              f"forest {np.mean((rs - (y[te][c] == 2)) ** 2):.4f}")
    if args.export:
        final = fit(X, y)          # train + test opponents for the shipped model
        export(final, args.export)


if __name__ == "__main__":
    main()

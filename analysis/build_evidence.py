# -*- coding: utf-8 -*-
"""
build_evidence.py - executes the code cells of the four Session 4 notebooks on the local data and
freezes every number the slides and the notebook texts quote into evidence/results.json.

The notebooks are the single source: the code cells come from scripts/build_notebooks.py and run
here exactly as written (the data address points at the local data/ folder). A few slide-only numbers
(the tiny split example, single trees of the forest, boosting step by step, the promotion table, local
against global models) are computed below from the same trained objects.

    python3 analysis/build_evidence.py        # about 30 seconds
"""
import contextlib
import io
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_notebooks as B  # noqa: E402

warnings.filterwarnings("ignore")
plt.show = lambda *a, **k: plt.close("all")


def _knob(m):
    try:
        return str(B.lookup(R, m.group(1)))
    except KeyError:                       # notebook 2 itself, before its choice exists: only a comment, leave it
        return m.group(0)


def run(n):
    """Execute notebook n's code cells in one namespace, from the course folder, on the local data."""
    ns = {"__name__": f"notebook{n}"}
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for i, src in enumerate(B.code_cells(n)):
            src = src.replace(B.RAW, "data")
            src = re.sub(r"\{\{(nb2\.chosen\.[a-z]+)\}\}", _knob, src)        # the knobs notebook 2 chose (after it has run)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"notebook {n}, code cell {i + 1}", "exec"), ns)
    finally:
        os.chdir(cwd)
        plt.close("all")
    return ns


def f(x, d=1):
    return round(float(x), d)


def score(actual, forecast):
    miss = np.asarray(forecast, float) - np.asarray(actual, float)
    return {"MAE": f(np.abs(miss).mean()), "WAPE": f(np.abs(miss).sum() / np.asarray(actual, float).sum() * 100),
            "bias": f(miss.mean())}


t0 = time.time()
R = {}

# ------------------------------------------------------------------ the data itself
sales = pd.read_csv(ROOT / "data/drinks_weekly.csv", parse_dates=["week_start"])
products = pd.read_csv(ROOT / "data/products.csv")
weeks = pd.read_csv(ROOT / "data/weeks.csv", parse_dates=["week_start"])
yr = sales["week_start"].dt.year
R["data"] = {
    "rows": len(sales), "columns": sales.shape[1], "products": int(sales["product"].nunique()),
    "weeks": int(sales["week_start"].nunique()),
    "first_week": str(sales["week_start"].min().date()), "last_week": str(sales["week_start"].max().date()),
    "mean_water": int(round(sales.loc[sales["product"] == "Mineral water", "units"].mean())),
    "mean_mulled": int(round(sales.loc[sales["product"] == "Mulled wine", "units"].mean())),
    "max_mulled": int(sales.loc[sales["product"] == "Mulled wine", "units"].max()),
    "promo_weeks": int((sales["discount_pct"] > 0).sum()),
    "promo_share": f((sales["discount_pct"] > 0).mean() * 100),
    "holiday_weeks": int((weeks["holiday_days"] > 0).sum()),
    "total_cases": int(sales["units"].sum()),
    "temp_min": f(weeks["temp_actual"].min()), "temp_max": f(weeks["temp_actual"].max()),
    "product_list": list(products["product"]),
    "yearly": {p: {str(y): int(round(v)) for y, v in g.groupby(g["week_start"].dt.year)["units"].mean().items()}
               for p, g in sales.groupby("product", sort=False)},
    "categories": dict(zip(products["product"], products["category"])),
    "product_mean": {p: int(round(v)) for p, v in sales.groupby("product", sort=False)["units"].mean().items()},
    "cases": dict(zip(products["product"], products["case"])),
    "pallet": dict(zip(products["product"], products["cases_per_pallet"].astype(int))),
}
e = R["data"]["yearly"]["Energy drink"]
R["data"]["energy_growth"] = f((e["2025"] / e["2024"] - 1) * 100, 0)
w24 = sales[yr == 2024]
R["data"]["weekly_2024"] = {"weeks": [d.strftime("%d %b") for d in sorted(w24["week_start"].unique())]}
for p in ["Mineral water", "Ice tea", "Mulled wine", "Energy drink"]:
    R["data"]["weekly_2024"][p] = w24.loc[w24["product"] == p, "units"].astype(int).tolist()
row0 = sales.iloc[[0, 1]].copy()
R["data"]["example_rows"] = [{k: (str(v.date()) if hasattr(v, "date") else (v.item() if hasattr(v, "item") else v))
                              for k, v in r.items()} for r in row0.to_dict("records")]

# ------------------------------------------------------------------ notebook 1
n1 = run(1)
ice, line, tree, depths = n1["ice"], n1["line"], n1["tree"], n1["depths"]
t_ = tree.tree_
names = ["temp_forecast", "discount_pct"]


def node(i):
    if t_.children_left[i] == -1:
        return {"leaf": True, "value": f(t_.value[i][0][0], 2), "samples": int(t_.n_node_samples[i])}
    return {"leaf": False, "feature": names[t_.feature[i]], "threshold": f(t_.threshold[i], 2),
            "samples": int(t_.n_node_samples[i]), "value": f(t_.value[i][0][0], 2),
            "yes": node(t_.children_left[i]), "no": node(t_.children_right[i])}


struct = node(0)
assert struct["feature"] == "temp_forecast" and struct["no"]["feature"] == "discount_pct", "the depth-2 tree changed shape"
R["nb1"] = {
    "ice_weeks": len(ice), "ice_promo_weeks": int((ice["discount_pct"] > 0).sum()),
    "line_slope": f(line.params["temp_forecast"], 2), "line_intercept": f(line.params["Intercept"], 2),
    "line_at0": int(round(line.params["Intercept"])),
    "line_at30": int(round(line.params["Intercept"] + 30 * line.params["temp_forecast"])),
    "tree": {"structure": struct, "root_threshold": struct["threshold"],
             "cold_threshold": struct["yes"]["threshold"],
             "cold": struct["yes"]["yes"]["value"], "cold_n": struct["yes"]["yes"]["samples"],
             "mild": struct["yes"]["no"]["value"], "mild_n": struct["yes"]["no"]["samples"],
             "warm_no": struct["no"]["yes"]["value"], "warm_no_n": struct["no"]["yes"]["samples"],
             "warm_promo": struct["no"]["no"]["value"], "warm_promo_n": struct["no"]["no"]["samples"],
             "promo_threshold": struct["no"]["threshold"]},
    "n_learn": len(n1["learn"]), "n_check": len(n1["check"]), "n_test": len(n1["test"]), "n_train": len(n1["train"]),
    "compare": {m: score(n1["test"]["units"], n1["test"][m]) for m in n1["methods"]},
    "depths": {int(d): {"train": f(r.iloc[0]), "valid": f(r.iloc[1])} for d, r in depths.iterrows()},
    "best_depth": int(n1["best_depth"]), "best_depth_wape": f(depths.iloc[:, 1].min()),
    "depth_train_last": f(depths.iloc[-1, 0]), "depth_valid_last": f(depths.iloc[-1, 1]),
}
R["nb1"]["compare"]["Decision tree"]["depth"] = int(n1["best_depth"])
# the scatter and the two curves for the slides
R["nb1"]["scatter"] = {"temp": ice["temp_forecast"].round(1).tolist(), "units": ice["units"].astype(int).tolist(),
                       "promo": (ice["discount_pct"] > 0).astype(int).tolist()}
grid = np.round(np.arange(-5, 36, 0.5), 1)
g = pd.DataFrame({"temp_forecast": grid, "discount_pct": 0})
R["nb1"]["grid"] = {"temp": grid.tolist(), "line": [f(v) for v in line.predict(g)], "line2": [f(v) for v in n1["line2"].predict(g)],
                    "tree": [f(v) for v in tree.predict(g[names])]}
R["nb1"]["line2_at0"] = int(round(n1["line2"].predict(pd.DataFrame({"temp_forecast": [0.0], "discount_pct": [0]}))[0]))
tst = n1["test"]
R["nb1"]["series_2025"] = {"weeks": [d.strftime("%d %b") for d in tst["week_start"]],
                           "Actual sales": tst["units"].astype(int).tolist(),
                           **{m: [f(v, 0) for v in tst[m]] for m in n1["methods"]}}

# what a metric is: five real weeks of 2024, forecast by the tree of the depth chosen on 2024 (the slide after "Deeper Is Not Better")
from sklearn.tree import DecisionTreeRegressor as _DT  # noqa: E402
_lrn, _chk, _inp = n1["learn"], n1["check"], n1["inputs"]
_t3 = _DT(max_depth=int(n1["best_depth"]), random_state=42).fit(_lrn[_inp], _lrn["units"])
_c = _chk.assign(fc=_t3.predict(_chk[_inp]))
assert abs(score(_c["units"], _c["fc"])["WAPE"] - R["nb1"]["best_depth_wape"]) < 0.05, "the depth chosen on 2024 does not reproduce"
_w = _c[(_c["week_start"] >= "2024-05-27") & (_c["week_start"] <= "2024-06-24")]
_sold, _fc = _w["units"].astype(int).tolist(), [int(round(v)) for v in _w["fc"]]
_miss = [a - b for a, b in zip(_fc, _sold)]
R["metric_demo"] = {"weeks": [d.strftime("%d %b") for d in _w["week_start"]], "first": str(_w["week_start"].min().date()),
                    "last": str(_w["week_start"].max().date()), "depth": int(n1["best_depth"]),
                    "sold": _sold, "forecast": _fc, "miss": _miss, "size": [abs(m) for m in _miss],
                    "sold_sum": sum(_sold), "forecast_sum": sum(_fc), "size_sum": sum(abs(m) for m in _miss),
                    "wape": f(sum(abs(m) for m in _miss) / sum(_sold) * 100), "mae": f(sum(abs(m) for m in _miss) / len(_miss), 0),
                    "bias": f(sum(_miss) / len(_miss), 0), "per100": int(round(sum(abs(m) for m in _miss) / sum(_sold) * 100)),
                    "biggest_week": _w["week_start"].iloc[int(np.argmax([abs(m) for m in _miss]))].strftime("%d %b"),
                    "year_wape": R["nb1"]["best_depth_wape"]}

# underfitting and overfitting: ice tea weeks without a promotion, temperature only; learn 2023, check 2024
_q = n1["ice_all"][n1["ice_all"]["discount_pct"] == 0]
_qa, _qb = _q[_q["week_start"].dt.year == 2023], _q[_q["week_start"].dt.year == 2024]
_X = lambda d: d[["temp_forecast"]]
_new = {d: score(_qb["units"], _DT(max_depth=d, random_state=42).fit(_X(_qa), _qa["units"]).predict(_X(_qb)))["WAPE"] for d in range(1, 9)}
_best = min(_new, key=_new.get)
_grid = np.round(np.arange(-5, 35.001, 0.25), 2)
R["fit_demo"] = {"grid": _grid.tolist(), "points": [[f(a), int(b)] for a, b in zip(_qa["temp_forecast"], _qa["units"])],
                 "n_learn": len(_qa), "n_new": len(_qb), "best_depth": int(_best), "fits": {}}
for _key, _d in (("under", 1), ("right", int(_best)), ("over", None)):
    _m = _DT(max_depth=_d, random_state=42).fit(_X(_qa), _qa["units"])
    R["fit_demo"]["fits"][_key] = {"depth": _d, "leaves": int(_m.get_n_leaves()),
                                   "learn": score(_qa["units"], _m.predict(_X(_qa)))["WAPE"], "new": score(_qb["units"], _m.predict(_X(_qb)))["WAPE"],
                                   "curve": [f(v) for v in _m.predict(pd.DataFrame({"temp_forecast": _grid}))]}

# the tiny split example: six real weeks of ice tea without a promotion, spread over the temperatures
quiet = ice[ice["discount_pct"] == 0].sort_values("temp_forecast")
picks = []
for target in [6, 11, 15, 20, 25, 29]:
    i = (quiet["temp_forecast"] - target).abs().idxmin()
    picks.append(i)
    quiet = quiet.drop(i)
six = ice.loc[picks, ["week_start", "temp_forecast", "units"]].sort_values("temp_forecast")
temps, units = six["temp_forecast"].to_numpy(), six["units"].to_numpy().astype(float)
cands = []
for k in range(1, len(six)):
    thr = (temps[k - 1] + temps[k]) / 2
    left, right = units[:k], units[k:]
    sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
    mae = (np.abs(left - left.mean()).sum() + np.abs(right - right.mean()).sum()) / len(units)
    cands.append({"threshold": f(thr), "left_mean": int(round(left.mean())), "right_mean": int(round(right.mean())),
                  "left_n": int(k), "right_n": int(len(units) - k), "sse_thousands": int(round(sse / 1000)), "mae": int(round(mae))})
best = min(cands, key=lambda c: c["sse_thousands"])
R["tiny_split"] = {"weeks": [d.strftime("%d %b %Y") for d in six["week_start"]], "temp": [f(t) for t in temps],
                   "units": [int(u) for u in units], "candidates": cands, "best": best,
                   "mean_all": int(round(units.mean())),
                   "mae_all": int(round(np.abs(units - units.mean()).mean())),
                   "best_by_mae": min(cands, key=lambda c: c["mae"])["threshold"]}

# ------------------------------------------------------------------ notebook 2
n2 = run(2)
test, train = n2["test"], n2["train"]
M = n2["METHODS"]
_ch = n2["chosen"]
R["nb2"] = {"learn_rows": len(n2["learn_2023"]), "check_rows": len(n2["check_2024"]),
            "chosen": {"tree": _ch["Decision tree, max_depth"], "forest": _ch["Random forest, max_features"], "lgbm": _ch["LightGBM, num_leaves"]},
            "valid_errors": {k: {str(v): f(e) for v, e in d.items()} for k, d in n2["valid_errors"].items()},
            "train_rows": int(n2["X_train"].shape[0]), "test_rows": int(n2["X_test"].shape[0]),
            "n_columns": int(n2["X_train"].shape[1]), "methods": M,
            "results": {m: score(test["units"], test[m]) for m in M},
            "by_product": {p: {m: f(n2["by_product"].loc[p, m]) for m in M} for p in n2["by_product"].index},
            "train_from": str(train["week_start"].min().date()), "train_to": str(train["week_start"].max().date()),
            "test_from": str(test["week_start"].min().date()), "test_to": str(test["week_start"].max().date())}
for m, col in zip(M, n2["results"]["WAPE on 2023-24 (learned from), %"]):
    R["nb2"]["results"][m]["train_WAPE"] = None if pd.isna(col) else f(col)
R["nb2"]["total_test_cases"] = int(test["units"].sum())
R["nb2"]["local"] = {k: f(v) for k, v in n2["per_product"]["Regression per product"].items()}
R["nb2"]["per_product"] = {p: {m: f(n2["per_product"].loc[p, m]) for m in n2["side"]} for p in n2["per_product"].index}
for p in ["Ice tea", "Energy drink", "Radler", "Mineral water"]:
    one = test[test["product"] == p]
    R["nb2"].setdefault("series_2025", {})[p] = {"weeks": [d.strftime("%d %b") for d in one["week_start"]],
                                                 "Actual sales": one["units"].astype(int).tolist(),
                                                 **{m: [f(v, 0) for v in one[m]] for m in M}}
# one full row of inputs, as the model sees it
fr = test[(test["product"] == "Radler") & (test["week_start"] == "2025-06-23")].iloc[0]
R["nb2"]["feature_row"] = {k: (fr[k] if isinstance(fr[k], str) else f(fr[k], 1)) for k in n2["FEATURES"]}
R["nb2"]["feature_row"].update({"week_start": str(fr["week_start"].date()), "units": int(fr["units"])})
# the promotion table: sales against the recent level, cold drinks, the two training years
cold = ["Mineral water", "Ice tea", "Apple spritzer", "Lager beer", "Radler"]
pv = n2["sales"][(n2["sales"]["week_start"].dt.year.between(2023, 2024)) & n2["sales"]["product"].isin(cold)].copy()
pv["band"] = pd.cut(pv["temp_forecast"], [-50, 10, 20, 25, 50], labels=["below 10 °C", "10 to 20 °C", "20 to 25 °C", "25 °C and more"], right=False)
pv["promo"] = np.where(pv["discount_pct"] > 0, "promotion", "no promotion")
pv["ratio"] = pv["units"] / pv["avg_last_4"]
tab = pv.groupby(["band", "promo"], observed=True)["ratio"].agg(["mean", "size"]).unstack()
R["promo_table"] = {"bands": [str(b) for b in tab.index],
                    "no_promo": [f(v, 2) for v in tab[("mean", "no promotion")]],
                    "promo": [f(v, 2) for v in tab[("mean", "promotion")]],
                    "n_promo": [int(v) for v in tab[("size", "promotion")]],
                    "n_no": [int(v) for v in tab[("size", "no promotion")]],
                    "products": cold}
cool = pv[pv["temp_forecast"] < 20]
warm = pv[pv["temp_forecast"] >= 20]
_two = pv.assign(band2=np.where(pv["temp_forecast"] < 20, "below 20 °C", "20 °C and more"))
_t2 = _two.groupby(["band2", "promo"])["ratio"].agg(["mean", "size"]).unstack().reindex(["below 20 °C", "20 °C and more"])
R["promo_table2"] = {"bands": list(_t2.index), "no_promo": [f(v, 2) for v in _t2[("mean", "no promotion")]],
                     "promo": [f(v, 2) for v in _t2[("mean", "promotion")]],
                     "n_promo": [int(v) for v in _t2[("size", "promotion")]], "n_no": [int(v) for v in _t2[("size", "no promotion")]],
                     "lift": [f((_t2.loc[b_, ("mean", "promotion")] / _t2.loc[b_, ("mean", "no promotion")] - 1) * 100, 0) for b_ in _t2.index]}
R["promo_table"]["lift_cool"] = f((cool.loc[cool["promo"] == "promotion", "ratio"].mean() / cool.loc[cool["promo"] == "no promotion", "ratio"].mean() - 1) * 100, 0)
R["promo_table"]["lift_warm"] = f((warm.loc[warm["promo"] == "promotion", "ratio"].mean() / warm.loc[warm["promo"] == "no promotion", "ratio"].mean() - 1) * 100, 0)

# one week, many trees: the forest's single trees and boosting step by step
row = test[(test["product"] == "Radler") & (test["week_start"] == "2025-06-23")]
Xr = n2["encode"](row)
forest, boost = n2["forest"], n2["boost"]
per_tree = np.array([t.predict(Xr.to_numpy())[0] for t in forest.estimators_])
staged = {k: f(boost.predict(Xr, num_iteration=k)[0], 0) for k in [1, 2, 3, 10, 50, 100, 200, 400]}
R["one_week"] = {"product": "Radler", "week": "2025-06-23", "actual": int(row["units"].iloc[0]),
                 "temp": f(row["temp_forecast"].iloc[0]), "discount": int(row["discount_pct"].iloc[0]),
                 "trees_first5": [int(round(v)) for v in per_tree[:5]], "forest": int(round(per_tree.mean())),
                 "tree_min": int(round(per_tree.min())), "tree_max": int(round(per_tree.max())),
                 "n_trees": len(per_tree), "boost_steps": staged, "train_mean": int(round(n2["y_train"].mean())),
                 "lgbm": int(round(boost.predict(Xr)[0]))}

# local against global, for the reference slide
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
local = []
cols = [c for c in n2["FEATURES"] if c != "product"]
for p, d in n2["data"].groupby("product"):
    tr, te = d[d["week_start"].dt.year <= 2024], d[d["week_start"].dt.year == 2025]
    rf = RandomForestRegressor(n_estimators=300, max_features=R["nb2"]["chosen"]["forest"], random_state=42, n_jobs=-1).fit(tr[cols], tr["units"])
    lg = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=R["nb2"]["chosen"]["lgbm"], min_child_samples=10, random_state=42,
                       verbose=-1).fit(tr[cols], tr["units"])
    local.append(pd.DataFrame({"product": p, "units": te["units"].to_numpy(), "rf": rf.predict(te[cols]), "lgb": lg.predict(te[cols])}))
local = pd.concat(local)
R["local_global"] = {"Random forest": {"local": score(local["units"], local["rf"])["WAPE"], "global": R["nb2"]["results"]["Random forest"]["WAPE"]},
                     "LightGBM": {"local": score(local["units"], local["lgb"])["WAPE"], "global": R["nb2"]["results"]["LightGBM"]["WAPE"]},
                     "rows_local": int(len(train) / R["data"]["products"]), "rows_global": int(len(train))}

# ------------------------------------------------------------------ notebook 3
n3 = run(3)
t3 = n3["test"]
ai_imp = pd.Series(n3["model"].feature_importances_, index=n3["X"].columns).sort_values(ascending=False)
st = n3["steps"]
R["nb3"] = {"ai": {"wape": st["AI's code"], "top": ai_imp.index[0], "top_share": f(ai_imp.iloc[0], 2), "top_share_pct": f(ai_imp.iloc[0] * 100, 1),
                   "second": ai_imp.index[1], "steps": [st["AI's code"], st["without pallets and temp_actual"], st["... and split by time"]],
                   "n_test_rows": int(len(n3["y_te"]))}}
en = t3[t3["product"] == "Energy drink"]
R["nb3"]["energy"] = {"actual": int(round(en["units"].mean())), "forest": int(round(en["Random forest"].mean())),
                      "no_history": int(round(en["Forest without history"].mean())),
                      "train_max": int(n3["train"].loc[n3["train"]["product"] == "Energy drink", "units"].max()),
                      "weeks": [d.strftime("%d %b") for d in en["week_start"]], "series": {
                          "Actual sales": en["units"].astype(int).tolist(),
                          "Random forest": [f(v, 0) for v in en["Random forest"]],
                          "Random forest without history": [f(v, 0) for v in en["Forest without history"]]}}
hw = t3[t3["week_start"] == "2025-06-30"].set_index("product")
R["nb3"]["heat"] = {"week": "2025-06-30", "temp": f(hw.loc["Ice tea", "temp_forecast"]), "train_max": f(n3["train"]["temp_forecast"].max()),
                    "ice_actual": int(hw.loc["Ice tea", "units"]), "ice_lgbm": int(round(hw.loc["Ice tea", "LightGBM"])),
                    "ice_reg": int(round(hw.loc["Ice tea", "Linear regression"])), "ice_discount": int(hw.loc["Ice tea", "discount_pct"]),
                    "water_actual": int(hw.loc["Mineral water", "units"]), "water_lgbm": int(round(hw.loc["Mineral water", "LightGBM"])),
                    "water_reg": int(round(hw.loc["Mineral water", "Linear regression"]))}
imp = n3["relies"].sort_values(ascending=False)
R["nb3"]["importance"] = {"values": {k: f(v) for k, v in imp.items()}, "top3": ", ".join(imp.index[:3]),
                          "base_wape": R["nb2"]["results"]["LightGBM"]["WAPE"]}
# the temperature curves, recomputed exactly as the notebook cell does
curves = {}
for product, week in n3["weeks"].items():
    r_ = t3[(t3["product"] == product) & (t3["week_start"] == week)]
    wi = pd.concat([r_] * len(n3["temps"]), ignore_index=True)
    wi["temp_forecast"] = n3["temps"]
    curves[product] = {"week": week, "forecast": [f(v, 0) for v in n3["boost"].predict(n3["encode"](wi))],
                       "temp_this_week": f(r_["temp_forecast"].iloc[0])}
R["nb3"]["curves"] = {"temps": [int(t) for t in n3["temps"]], "products": curves, "reg_slope": f(n3["reg"].params["temp_forecast"], 2)}
_tx = []
for product, c in curves.items():
    fc = dict(zip(R["nb3"]["curves"]["temps"], c["forecast"]))
    _tx.append(f"{product.lower() if product != 'Radler' else 'radler'} from about {fc[10]:,.0f} cases at 10 °C to {fc[30]:,.0f} at 30 °C")
R["nb3"]["curves"]["text"] = "; ".join(_tx) + "."
parts = n3["parts"]
R["nb3"]["explain"] = {"product": n3["product"], "week": n3["week"], "forecast": f(parts.sum(), 0),
                       "actual": int(n3["row"]["units"].iloc[0]), "start": f(parts["reference: average forecast"], 0),
                       "parts": {k: f(v, 0) for k, v in parts.drop("reference: average forecast").sort_values(key=abs, ascending=False).items()},
                       "values": {k: (v if isinstance(v, str) else f(v, 1)) for k, v in n3["values"].items()},
                       "temp": f(n3["values"]["temp_forecast"]), "temp_part": f(parts["temp_forecast"], 0),
                       "discount": int(n3["values"]["discount_pct"]), "discount_part": f(parts["discount_pct"], 0)}
_pl = n3["plans"]
R["nb3"]["what_if"] = {"as_planned": int(round(n3["boost"].predict(n3["encode"](_pl["as planned"]))[0])),
                       "no_promo": int(round(n3["boost"].predict(n3["encode"](_pl["without the promotion"]))[0])),
                       "cool": int(round(n3["boost"].predict(n3["encode"](_pl["a cool week, 18 °C"]))[0]))}
# the regression's odd forecasts, for the slide on faulty outputs
t2 = n2["test"]
neg = t2.loc[t2["Linear regression"].idxmin()]
mw = t2[(t2["product"] == "Mulled wine") & (t2["week_start"].dt.month.isin([6, 7, 8]))]
R["faulty"] = {"reg_min": int(round(neg["Linear regression"])), "reg_min_product": neg["product"],
               "reg_min_week": str(neg["week_start"].date()), "reg_min_actual": int(neg["units"]),
               "reg_negative_weeks": int((t2["Linear regression"] < 0).sum()),
               "mulled_summer_reg": int(round(mw["Linear regression"].mean())), "mulled_summer_actual": int(round(mw["units"].mean())),
               "mulled_summer_lgbm": int(round(mw["LightGBM"].mean())),
               "tree_negative_weeks": int((t2[["Decision tree", "Random forest"]] < 0).sum().sum()),
               "lgbm_negative_weeks": int((t2["LightGBM"] < 0).sum()),
               "lgbm_min": int(round(t2["LightGBM"].min())), "lgbm_min_product": t2.loc[t2["LightGBM"].idxmin(), "product"],
               "lgbm_min_week": str(t2.loc[t2["LightGBM"].idxmin(), "week_start"].date()),
               "lgbm_min_actual": int(t2.loc[t2["LightGBM"].idxmin(), "units"])}

# ------------------------------------------------------------------ notebook 4 (the project template on the course data)
n4 = run(4)
res4 = n4["results"]
_d4 = n4["dates"]
R["nb4"] = {"fit_rows": int(n4["is_fit"].sum()), "fit_from": str(_d4[n4["is_fit"]].min().date()),
            "valid_rows": int(n4["is_valid"].sum()), "valid_from": str(_d4[n4["is_valid"]].min().date()),
            "test_rows": int(n4["is_test"].sum()), "test_from": str(_d4[n4["is_test"]].min().date()),
            "chosen": n4["chosen_method"], "valid_wape": {m: f(v) for m, v in n4["validation"]["WAPE, %"].items()},
            "valid_wape_chosen": f(n4["validation"].loc[n4["chosen_method"], "WAPE, %"]),
            "valid_lgbm": f(n4["validation"].loc["LightGBM", "WAPE, %"]),
            "test_wape": {m: f(v) for m, v in n4["results"]["WAPE, %"].items()},
            "experiment_wape": f(score(n4["y_all"][n4["is_valid"]], n4["policy"](n4["exp"].predict(n4["X_exp"][n4["is_valid"]])))["WAPE"]),
            "next_week": str(n4["next_period"]["week_start"].iloc[0].date()),
            "next_water": float(n4["next_period"].set_index("product").loc["Mineral water", "forecast"])}

# ------------------------------------------------------------------ software, for the record
import sklearn, lightgbm, statsmodels  # noqa: E402,E401
R["software"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                 "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "statsmodels": statsmodels.__version__}

out = ROOT / "evidence" / "results.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"evidence/results.json written in {time.time() - t0:.0f} s")
for m, v in R["nb2"]["results"].items():
    print(f"  {m:20s} WAPE {v['WAPE']:5.1f} %   MAE {v['MAE']:6.1f}   bias {v['bias']:6.1f}   seen {v['train_WAPE']}")
print("  AI steps", R["nb3"]["ai"]["steps"], "| best depth", R["nb1"]["best_depth"], "| tiny split", R["tiny_split"]["best"]["threshold"])

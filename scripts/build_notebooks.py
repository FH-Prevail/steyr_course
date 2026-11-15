# -*- coding: utf-8 -*-
"""
build_notebooks.py - the four Colab notebooks of Session 4, "Forecasting With Tree-Based Models"
(Advanced Data Analysis Techniques, Master Supply Chain Management, FH OÖ Steyr).

Every notebook is written from the cell lists below. The code cells are the single source of
the numbers: analysis/build_evidence.py executes exactly these cells on the local data and freezes
what they print into evidence/results.json; this script then fills the {{...}} placeholders in the
markdown and in the # CHECK comments from that file, so a CHECK value can never drift from what the
cell really prints. scripts/verify_notebooks.py runs the finished notebooks and saves their outputs.

    python3 analysis/build_evidence.py      # executes the cells, writes evidence/results.json
    python3 scripts/build_notebooks.py      # writes notebooks/*.ipynb (without outputs)
    python3 scripts/verify_notebooks.py     # runs them, saves outputs, checks the CHECK values
"""
import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
REPO = "FH-Prevail/steyr_course"                       # the public repository (student material only)
RAW = f"https://raw.githubusercontent.com/{REPO}/main/data"
COURSE = "Advanced Data Analysis Techniques · Session 4: Forecasting With Tree-Based Models · Sina Mirshahi, FH OÖ Steyr"
NB_NAMES = {1: "1_line_to_tree.ipynb", 2: "2_forest_and_boosting.ipynb", 3: "3_check_and_explain.ipynb",
            4: "4_project_template.ipynb"}
NB_TITLES = {1: "1 · From A Straight Line To A Tree", 2: "2 · Many Trees: Forest And Boosting",
             3: "3 · Check And Explain", 4: "4 · Project Template: Forecasting With Trees"}


def colab_url(n):
    return f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{NB_NAMES[n]}"


def md(text):
    return ("markdown", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def header(n, intro):
    url = colab_url(n)
    return md(f"""
# {NB_TITLES[n]}

<a href="{url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

*{COURSE}*

{intro}

**How to work with this notebook**

1. In Colab choose **File, Save a copy in Drive**. Your copy keeps your changes; this original never changes.
2. Run the cells from the top with **Shift + Enter**, or all at once with **Runtime, Run all**.
3. Read the result under each cell before you go on. You do not need to write code from scratch.

| Label | Meaning |
|---|---|
| `# EDIT` | a setting you may change, then run the cell again |
| `# CHECK` | the result for the unchanged course data; after an EDIT the numbers change, and that is fine |
| **Ask an AI** | a prompt that writes the same code in Gemini, ChatGPT or Claude, and how to check its answer |
| **Your answer** | a few words of your own: double-click the cell, write, then Shift + Enter |

The data is synthetic: a drinks wholesaler in Upper Austria, built to look like real weekly sales. No company data is contained.
""")


FIRST = code(f'''
# Start here: run this cell first. It loads the course data (one small table) and sets up the charts.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "{RAW}"   # the course data on GitHub
if Path("data/drinks_weekly.csv").exists():      # a copy next to the notebook (Google Drive or upload) is used first
    DATA = "data"
sales = pd.read_csv(f"{{DATA}}/drinks_weekly.csv", parse_dates=["week_start"])
sales["product"] = sales["product"].astype(object)   # plain text labels: the regression formula works with every pandas version

# one colour per method, the same as on the slides
COLOURS = {{"Actual sales": "#262626", "Repeat last year": "#E66A1F", "Linear regression": "#1B9E5A",
           "Decision tree": "#7A5C00", "Random forest": "#D04F95", "LightGBM": "#0B5CAD"}}
plt.rcParams.update({{"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False}})
pd.set_option("display.float_format", "{{:,.1f}}".format)
print("Ready:", len(sales), "rows,", sales["product"].nunique(), "products, weeks from",
      sales["week_start"].min().date(), "to", sales["week_start"].max().date())
# CHECK: Ready: {{{{data.rows}}}} rows, {{{{data.products}}}} products, weeks from {{{{data.first_week}}}} to {{{{data.last_week}}}}
''')

SCORE_SRC = '''
def score(actual, forecast):
    """Three numbers for a forecast. MAE: the average miss in cases. WAPE: all misses added up, as a
    percentage of all cases sold. Bias: forecast minus actual, on average (minus means too low)."""
    if isinstance(actual, pd.Series) and isinstance(forecast, pd.Series):
        assert actual.index.equals(forecast.index), "Actuals and forecasts must belong to the same weeks."
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    assert a.ndim == 1 and a.size > 0 and a.shape == f.shape, "One forecast for every week, in one column."
    assert np.isfinite(a).all() and np.isfinite(f).all(), "Every week needs a number: no missing actual or forecast."
    miss, total = f - a, np.abs(a).sum()
    return pd.Series({"MAE, cases": np.abs(miss).mean(),
                      "WAPE, %": np.abs(miss).sum() / total * 100 if total else np.nan,   # undefined if nothing was sold
                      "Bias, cases": miss.mean()})
'''.strip("\n")
SCORE = code(SCORE_SRC)

CONTEXT = md('''
> **Ask an AI · the context first.** In ChatGPT or Claude, paste this before any prompt of this notebook (Gemini in Colab already sees the notebook):
>
> *"I work in Google Colab with pandas, scikit-learn, statsmodels and LightGBM. My table `sales` has one row per product and week: `week_start` (the Monday), `product` (8 drinks), `units` (cases sold, the target), `pallets` (pallets picked in the warehouse, known only after the week), `temp_forecast` (the weather forecast for the week, mean daily maximum in °C), `temp_actual` (the weather that came, known only after the week), `discount_pct` (the planned promotion discount, 0 = none) and `holiday_days` (public holidays from Monday to Friday). We plan on Sunday evening for the coming week. Keep the code short and commented, and never use information that is unknown on that Sunday."*
>
> **Check every answer:** run it, compare with the `# CHECK` line, and make sure you can say what each line does. Share only course data with an AI tool, never company data.
''')

# ============================================================================================ notebook 1
NB1 = [
    header(1, """
**Hands-on 1.** One product, ice tea, and two ways to describe it: the straight line you know from session 2, and a decision tree. You will

- look at the data and see how ice tea reacts to the temperature,
- fit the straight line and find where it goes wrong,
- grow your first decision tree and read its rules,
- test both on 2025, a year they have never seen, and find out how deep a tree should grow.
"""),
    FIRST,
    CONTEXT,
    md("""
## 1. The data

One row is one product in one week: {{data.products}} products × {{data.weeks}} weeks = {{data.rows}} rows. We plan on **Sunday evening** for the week that starts on Monday, so the last column says what we know at that moment.

| Column | What it is | Known on Sunday evening? |
|---|---|---|
| `week_start` | the Monday of the week | yes |
| `product` | one of eight drinks | yes |
| `units` | cases sold in the week: **the number we forecast** (the target) | no, it is the answer |
| `pallets` | pallets picked in the warehouse for these cases | no, only after the week |
| `temp_forecast` | the weather forecast for the week: mean of the daily maximum, °C | yes |
| `temp_actual` | the weather that came | no, only after the week |
| `discount_pct` | the planned promotion discount, 0 = no promotion | yes, promotions are planned |
| `holiday_days` | public holidays from Monday to Friday | yes |
"""),
    code('''
sales.head(8)
'''),
    code('''
sales.groupby("product")["units"].agg(["mean", "min", "max"]).round(0).astype(int)
# CHECK: Mineral water sells about {{data.mean_water|,}} cases in an average week
'''),
    md("""
**What to see.** The eight products live on very different scales: mineral water sells about {{data.mean_water|,}} cases in an average week, mulled wine about {{data.mean_mulled|,}}, but up to {{data.max_mulled|,}} in a week of its season.

## 2. Look before you model

A model can only find what is in the data. Look first.
"""),
    code('''
show = ["Mineral water", "Ice tea", "Mulled wine"]   # EDIT: any of the eight products
year = 2024                                         # EDIT: 2022 to 2025

fig, ax = plt.subplots(figsize=(10, 3.8))
for product in show:
    one = sales[(sales["product"] == product) & (sales["week_start"].dt.year == year)]
    ax.plot(one["week_start"], one["units"], marker="o", ms=3, label=product)
ax.set(title=f"Weekly sales in {year}", ylabel="cases per week")
ax.legend()
plt.show()
'''),
    md("""
**What to see.** Water and ice tea peak in summer; mulled wine sells only in Advent and almost nothing the rest of the year. Each product follows its own conditions.

Now ice tea against the temperature forecast, in the two years we learn from (2023 and 2024). Each dot is one week.
"""),
    code('''
ice = sales[(sales["product"] == "Ice tea") & (sales["week_start"].dt.year.between(2023, 2024))].copy()
promo = ice["discount_pct"] > 0

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(ice.loc[~promo, "temp_forecast"], ice.loc[~promo, "units"], s=18, color="#0B5CAD", label="no promotion")
ax.scatter(ice.loc[promo, "temp_forecast"], ice.loc[promo, "units"], s=34, marker="^", color="#E66A1F", label="promotion")
ax.set(xlabel="temperature forecast for the week, °C", ylabel="cases of ice tea", title="Ice tea, 2023 and 2024")
ax.legend()
plt.show()
print(len(ice), "weeks,", promo.sum(), "of them with a promotion")
# CHECK: {{nb1.ice_weeks}} weeks, {{nb1.ice_promo_weeks}} of them with a promotion
'''),
    md("""
**What to see.** Below about 15 °C ice tea hardly reacts to the weather. Above 20 °C it climbs steeply. The triangles, promotion weeks, sit higher, and much higher in warm weeks. Keep this curve in mind: it is the shape a straight line cannot follow.

## 3. The straight line from session 2

The same simple regression as in session 2: one input, one straight line, fitted with ordinary least squares (OLS).

> **Ask an AI** · *"With statsmodels formula API, fit an OLS regression of units on temp_forecast for the DataFrame ice, store it as line, and print the coefficients rounded to 1 decimal."*
> **Check:** the slope must be the `# CHECK` value; if the AI uses scikit-learn instead, the numbers are the same.
"""),
    code('''
import statsmodels.formula.api as smf

line = smf.ols("units ~ temp_forecast", data=ice).fit()     # the simple regression of session 2
print(line.params.round(1))
print("The line at 0 °C: ", round(line.params["Intercept"]), "cases")
# CHECK: temp_forecast {{nb1.line_slope|.1f}}: one degree warmer, about {{nb1.line_slope|.0f}} cases more, in January as in July
# CHECK: at 0 °C the line says {{nb1.line_at0}} cases, fewer than nothing
'''),
    code('''
grid = pd.DataFrame({"temp_forecast": np.arange(-5, 36, 0.5), "discount_pct": 0})    # every temperature, no promotion

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(ice["temp_forecast"], ice["units"], s=15, color="#B0B0B0", label="weeks, 2023 and 2024")
ax.plot(grid["temp_forecast"], line.predict(grid), color=COLOURS["Linear regression"], lw=2.5, label="straight line")
ax.axhline(0, color="black", lw=0.8)
ax.set(xlabel="temperature forecast, °C", ylabel="cases of ice tea", title="One slope for every temperature")
ax.legend()
plt.show()
'''),
    md("""
**What to see.** The line is too high between about 5 and 18 °C, too low in hot weeks, and below zero in frost. One slope for all temperatures is the problem: how much one more degree adds depends on how warm it already is.

## 4. Your first decision tree

A **decision tree** asks yes-or-no questions about the inputs, one after the other, for example *"Is the temperature forecast above 20 °C?"*. The answers lead down to a **leaf**, and every leaf predicts the **average of the training weeks** that ended there.

Two settings, called knobs from now on:

- `max_depth=2`: at most two questions in a row,
- `min_samples_leaf=5`: every leaf must hold at least 5 weeks, so no rule rests on one odd week.

> **Ask an AI** · *"Fit a scikit-learn DecisionTreeRegressor with max_depth=2, min_samples_leaf=5 and random_state=42 on the columns temp_forecast and discount_pct of ice, target units, and print its rules with export_text."*
> **Check:** the first question must be the one in the `# CHECK` line.
"""),
    code('''
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

inputs = ["temp_forecast", "discount_pct"]
tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=42)    # EDIT: try max_depth=3
tree.fit(ice[inputs], ice["units"])                       # learn the questions from 2023 and 2024
print(export_text(tree, feature_names=inputs))
# CHECK: the first question is temp_forecast <= {{nb1.tree.root_threshold|.2f}}; a warm week with a promotion: {{nb1.tree.warm_promo|.2f}} cases
'''),
    code('''
plt.figure(figsize=(11, 4.6))
plot_tree(tree, feature_names=inputs, filled=True, impurity=False, rounded=True, precision=2, fontsize=10)
plt.show()
'''),
    md("""
**How to read it.** Start at the top. If the answer is yes (true), go left; if no, go right. `samples` is the number of training weeks in a box, `value` their average sales.

- A cool week (forecast {{nb1.tree.root_threshold|.1f}} °C or less) splits once more by temperature: {{nb1.tree.cold|.0f}} or {{nb1.tree.mild|.0f}} cases.
- A warm week splits by the promotion: {{nb1.tree.warm_no|,.0f}} cases without, {{nb1.tree.warm_promo|,.0f}} with one (`discount_pct <= 5` means no promotion).

**Notice:** the promotion question appears only on the warm side. In warm weeks the promotion makes the big difference; in cool weeks the temperature matters more. Nobody told the tree; it found the interaction in the data. Read any path as a rule: *if the forecast is above {{nb1.tree.root_threshold|.2f}} °C and a promotion is planned, expect about {{nb1.tree.warm_promo|,.0f}} cases.*
"""),
    code('''
line2 = smf.ols("units ~ temp_forecast + discount_pct", data=ice).fit()      # the same two inputs as the tree
fig, ax = plt.subplots(figsize=(7, 4.2))
no_promo = ice[ice["discount_pct"] == 0]
ax.scatter(no_promo["temp_forecast"], no_promo["units"], s=15, color="#B0B0B0", label="weeks without a promotion")
ax.plot(grid["temp_forecast"], line2.predict(grid), color=COLOURS["Linear regression"], lw=2.5,
        label="straight line, same two inputs")
ax.plot(grid["temp_forecast"], tree.predict(grid[inputs]), color=COLOURS["Decision tree"], lw=2.5,
        drawstyle="steps-post", label="decision tree, depth 2")
ax.set(xlabel="temperature forecast, °C", ylabel="cases of ice tea", title="Steps instead of a line")
ax.legend()
plt.show()
'''),
    md("""
**What to see.** Both models know the temperature and the promotion; here both are drawn for weeks without a promotion. The tree draws **steps**: flat, a jump, flat again. A tree forecasts averages of real weeks, so it never forecasts less than zero, while the line does. But it also never goes above its highest step: however hot it gets, the forecast stays where the hottest training weeks were. Remember this for later.

## 5. Learn, choose, test: three periods

A forecast must be judged on weeks the model has never seen, and every choice, such as the depth of a tree, must be made **without** looking at those weeks. So we use three periods:

| Period | What happens |
|---|---|
| 2023 | the tree learns |
| 2024 | we choose the depth: the **validation** year |
| 2025 | the final test, used once, after every choice is made |

Once the depth is chosen, the tree learns again from 2023 and 2024 together, and only then do we look at 2025.

**How we forecast:** one week at a time. Every Sunday evening we forecast the coming week with what is known that evening: the weather forecast, the promotion plan, the holidays. The final model is fitted on Sunday 5 January 2025, after the week that started on 30 December has finished (we assume its sales are final that evening), and then stays fixed. Two more inputs join now: the holidays and the week of the year (1 to 52 in our years).

Three numbers judge a forecast:

- **MAE**, the average miss, in cases: forecast 900, actual 1,000, miss 100.
- **WAPE**, all misses added up (too high and too low alike), divided by all cases actually sold: misses of 100 and 300 on sales of 1,000 and 3,000 give 400 ÷ 4,000 = 10 %. It is not a share of lost sales.
- **Bias**, forecast minus actual, on average: minus means the forecast was too low.
"""),
    SCORE,
    code('''
ice_all = sales[sales["product"] == "Ice tea"].sort_values("week_start").copy()
assert (ice_all["week_start"].diff().dropna() == pd.Timedelta(days=7)).all(), "A week is missing or doubled: shift(52) would not be 52 weeks back."
ice_all["week_of_year"] = ice_all["week_start"].dt.isocalendar().week.astype(int)
ice_all["same_week_last_year"] = ice_all["units"].shift(52)             # sales 52 weeks earlier
inputs = ["temp_forecast", "discount_pct", "holiday_days", "week_of_year"]

year = ice_all["week_start"].dt.year
learn, check, test = ice_all[year == 2023], ice_all[year == 2024], ice_all[year == 2025].copy()
print(len(learn), "weeks to learn,", len(check), "weeks to choose,", len(test), "weeks to test")
# CHECK: {{nb1.n_learn}} weeks to learn, {{nb1.n_check}} weeks to choose, {{nb1.n_test}} weeks to test
'''),
    md("""
**Choose the depth on 2024.** A deeper tree asks more questions and fits the weeks it learns from better. Does it also forecast new weeks better? We grow trees of depth 1 to 10, with nothing that stops them (no `min_samples_leaf`), learn from 2023, and score each on 2023 and on 2024.

> **Ask an AI** · *"For max_depth from 1 to 10, fit a DecisionTreeRegressor(random_state=42) on learn[inputs] and learn['units'], put the WAPE on learn and on check into one table and one line chart, and print the depth with the lowest WAPE on check."*
> **Check:** the depth it prints must be the one in the `# CHECK` line; the AI must never use `test` here.
"""),
    code('''
rows = []
for depth in range(1, 11):
    t = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(learn[inputs], learn["units"])
    rows.append({"depth": depth,
                 "WAPE on 2023 (learned from)": score(learn["units"], t.predict(learn[inputs]))["WAPE, %"],
                 "WAPE on 2024 (validation)": score(check["units"], t.predict(check[inputs]))["WAPE, %"]})
depths = pd.DataFrame(rows).set_index("depth")
best_depth = int(depths["WAPE on 2024 (validation)"].idxmin())
depths.plot(marker="o", figsize=(7, 3.8), color=["#0B5CAD", "#E66A1F"], ylabel="WAPE, %",
            title="Ice tea: error by tree depth")
plt.show()
print("The depth with the lowest error on 2024:", best_depth)
depths.round(1)
# CHECK: the depth with the lowest error on 2024: {{nb1.best_depth}} ({{nb1.best_depth_wape|.1f}} %); on 2023 the error falls to {{nb1.depth_train_last|.1f}} % at depth 10
'''),
    md("""
**What to see.** On the weeks it learned from, the error falls towards zero: a deep tree memorises 2023, odd weeks included (with this data it falls at every step, but that is not guaranteed). On 2024 the error is lowest at depth {{nb1.best_depth}}; more depth does not help any more. That is **overfitting**: the tree learns the noise of 2023 instead of the pattern. We choose depth {{nb1.best_depth}}, and we have not looked at 2025 yet.

## 6. The final test on 2025

The tree with the chosen depth learns again, from 2023 and 2024, and forecasts every week of 2025. Next to it: the straight line with the same four inputs, and the benchmark, which repeats last year (the sales 52 weeks earlier).
"""),
    code('''
train = ice_all[year.isin([2023, 2024])]                                  # learn again, from both years
test["Repeat last year"] = test["same_week_last_year"]                     # the benchmark
reg = smf.ols("units ~ temp_forecast + discount_pct + holiday_days + week_of_year", data=train).fit()
test["Linear regression"] = reg.predict(test)
final_tree = DecisionTreeRegressor(max_depth=best_depth, random_state=42).fit(train[inputs], train["units"])
test["Decision tree"] = final_tree.predict(test[inputs])

methods = ["Repeat last year", "Linear regression", "Decision tree"]
pd.DataFrame({m: score(test["units"], test[m]) for m in methods}).T
# CHECK: WAPE {{nb1.compare.Repeat last year.WAPE|.1f}} %, {{nb1.compare.Linear regression.WAPE|.1f}} % and {{nb1.compare.Decision tree.WAPE|.1f}} %
'''),
    code('''
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(test["week_start"], test["units"], color=COLOURS["Actual sales"], lw=2, marker="o", ms=3, label="Actual sales")
for m in methods:
    ax.plot(test["week_start"], test[m], color=COLOURS[m], lw=1.8, label=m)
ax.set(title="Ice tea in 2025: three forecasts against what sold", ylabel="cases per week")
ax.legend(ncol=4, fontsize=9)
plt.show()
'''),
    md("""
**What to see.** Repeating last year copies last summer's weather into this summer. The straight line misses the peaks and runs too high in winter. The tree follows the summer much better, with the steps you saw before. In our data, on this product, the tree wins; whether it wins on yours, only such a test can tell.

### Your answer (not graded; it prepares your project)

1. Read one rule from your depth-2 tree in section 4: *in a week warmer than ___ °C with a promotion, the tree expects about ___ cases.*
2. Which depth gives the lowest error on 2024? ___ Why do we choose the depth on 2024 and not on 2025? ___
3. In one sentence: why can the tree follow the ice tea curve better than the straight line?

**Next:** one tree is easy to read but rough. In notebook 2 many trees work together, for all eight products at once.
"""),
]

# ============================================================================================ notebook 2
NB2 = [
    header(2, """
**Hands-on 2.** All eight products at once, with the history of each product as input, and two ways to combine many trees. You will

- turn each product's past sales into inputs: last week, the last four weeks, the same week last year,
- split time into three periods: learn (2023), choose (2024), test (2025),
- choose the main knob of each tree model on 2024,
- test a decision tree, a random forest and LightGBM once on 2025, against the benchmark and the regression of session 2,
- find out for which products the trees pay off, and for which a regression is enough.

Sections 1 to 6 are the core; section 7 is optional.
"""),
    FIRST,
    CONTEXT,
    md("""
## 1. Inputs from the history

A tree cannot look back in time by itself. We give each week three numbers from the past, each computed **per product**:

- `last_week`: the sales one row, and here one week, back,
- `avg_last_4`: the average of the four weeks before,
- `same_week_last_year`: the sales 52 weeks back.

`shift(1)` moves every value one row down, so each week sees the row before it. That is the previous **week** only because our table has every week exactly once, sorted by date within each product; check this in your own data. `shift(-1)` would move next week's sales in: information from the future, called **leakage**. Then `week_of_year` (1 to 52 in our years) from the calendar.
"""),
    code('''
sales = sales.sort_values(["product", "week_start"]).reset_index(drop=True)
for product, part in sales.groupby("product"):                                  # every Monday once, no gaps
    expected = pd.date_range(part["week_start"].min(), part["week_start"].max(), freq="W-MON")
    assert len(part) == len(expected) and (part["week_start"].to_numpy() == expected.to_numpy()).all(), product
by_product = sales.groupby("product")["units"]
sales["last_week"] = by_product.shift(1)                                         # one week back
sales["avg_last_4"] = by_product.transform(lambda s: s.shift(1).rolling(4).mean())   # the 4 weeks before
sales["same_week_last_year"] = by_product.shift(52)                             # 52 weeks back
sales["week_of_year"] = sales["week_start"].dt.isocalendar().week.astype(int)

sales[sales["product"] == "Ice tea"][["week_start", "units", "last_week", "avg_last_4", "same_week_last_year"]].iloc[60:66]
# CHECK: in every row, last_week is the units of the row above
'''),
    md("""
**The inputs we use**, all known on Sunday evening: which product, the week of the year, the temperature forecast, the planned discount, the holidays, and the three history numbers. We leave out `pallets` and `temp_actual`: both are known only after the week.

## 2. Three periods: learn, choose, test

In session 2 the rows were companies, and a random 80/20 split was right. Here the rows are **weeks**: a random split would let the model learn from July 2025 to forecast June 2025, which no planner can do. So we split by time, into three periods:

- **2023**, to learn while we choose the knobs, and **2024**, to choose them (the validation year),
- **2023 and 2024** together, to learn the final models,
- **2025**, the final test, used once.

2022 only feeds the history numbers. **How we forecast:** the final models are trained once, on Sunday 5 January 2025, and then stay fixed. Every Sunday evening of 2025 they forecast the coming week, with that week's weather forecast and plan and the sales up to the week just finished. So the history inputs of a 2025 week use earlier 2025 sales: that is allowed, because they are known by then.

The comparisons on 2025 are planned now, before anything is tested: the benchmark, the regression of session 2, three tree models, and a regression per product. The trees need numbers, so `encode` turns the product into eight 0/1 columns, one per product.

> **Ask an AI** · *"Using `sales` after the history columns have been created, keep the rows with a complete history (drop rows without same_week_last_year). Use only product, week_of_year, temp_forecast, discount_pct, holiday_days, last_week, avg_last_4 and same_week_last_year as inputs; units is the target. Make four tables by date: learn_2023 and check_2024 for choosing settings, train (2023 and 2024) and test (2025). One-hot encode product with the same columns in every table. Print the shapes and the date ranges, and check that train ends before test begins."*
> **Check:** no random `train_test_split`; the shapes and dates must match the `# CHECK` line.
"""),
    code('''
FEATURES = ["product", "week_of_year", "temp_forecast", "discount_pct", "holiday_days",
            "last_week", "avg_last_4", "same_week_last_year"]
PRODUCTS = sorted(sales["product"].unique())
COLUMNS = FEATURES[1:] + ["product_" + p.replace(" ", "_") for p in PRODUCTS]


def encode(table):
    """The model inputs as numbers: the inputs above, plus one 0/1 column per product."""
    X = pd.get_dummies(table[FEATURES], columns=["product"], dtype=int)
    X.columns = [c.replace(" ", "_") for c in X.columns]
    return X.reindex(columns=COLUMNS, fill_value=0)


data = sales.dropna(subset=["same_week_last_year"])               # 2022 only feeds the history numbers
year = data["week_start"].dt.year
learn_2023, check_2024 = data[year == 2023], data[year == 2024]   # to choose the knobs
train, test = data[year <= 2024], data[year == 2025].copy()       # the final models, and the final test
X_train, y_train = encode(train), train["units"]
X_test, y_test = encode(test), test["units"]
print("learn:", len(learn_2023), "rows | choose:", len(check_2024), "rows | train:", X_train.shape, "| test:", X_test.shape)
print("train weeks:", train["week_start"].min().date(), "to", train["week_start"].max().date())
print("test weeks: ", test["week_start"].min().date(), "to", test["week_start"].max().date())
# CHECK: learn: {{nb2.learn_rows}} rows | choose: {{nb2.check_rows}} rows | train: ({{nb2.train_rows}}, {{nb2.n_columns}}) | test: ({{nb2.test_rows}}, {{nb2.n_columns}})
'''),
    SCORE,
    md("""
## 3. The benchmark and the straight line

The benchmark repeats last year: the sales 52 weeks earlier. The regression is the multiple regression of session 2 with all the inputs above; `C(product)` gives each product its own level, as `C(sector)` did for the sectors. It has no knobs to choose, so it learns from 2023 and 2024 directly. Their results on 2025 come in section 5, together with the trees, once every choice is made.
"""),
    code('''
import statsmodels.formula.api as smf

test["Repeat last year"] = test["same_week_last_year"]
reg = smf.ols("units ~ C(product) + week_of_year + temp_forecast + discount_pct + holiday_days"
              " + last_week + avg_last_4 + same_week_last_year", data=train).fit()
test["Linear regression"] = reg.predict(test)
print("The regression learned from", len(train), "rows; its result on 2025 comes in section 5.")
# CHECK: The regression learned from {{nb2.train_rows}} rows
'''),
    md("""
## 4. Three tree models, and their knobs chosen on 2024

- **Decision tree**: one tree, as in notebook 1, now for all products. Its main knob: `max_depth`.
- **Random forest**: 300 trees. Each tree learns from a random sample of the weeks, drawn with replacement (some weeks twice, some not at all), and at every question it may look at only a share of the inputs (`max_features`). The forest forecasts the **average** of its trees.
- **LightGBM**: boosting. The trees are built one after the other; each new tree learns what the trees before it still get wrong, and a small part of that correction (`learning_rate=0.05`) is added. Its main size knob: `num_leaves`.

All three learn from all eight products at once (global mode): what a promotion does to ice tea helps the model with radler.

**The rule:** for each model, try three values of its main knob, learn from 2023, score on 2024, and keep the value with the lowest error. 2025 stays untouched. (If Colab ever reports `ModuleNotFoundError: lightgbm`, run `!pip install lightgbm` in a new cell, then this cell again.)

> **Ask an AI** · *"For a scikit-learn DecisionTreeRegressor (max_depth 4, 6, 8, min_samples_leaf=5), a RandomForestRegressor (n_estimators=300, max_features 0.33, 0.5, 1.0) and a LightGBM LGBMRegressor (n_estimators=400, learning_rate=0.05, min_child_samples=10, num_leaves 7, 15, 31), all with random_state=42: fit each candidate on encode(learn_2023), compute the WAPE on encode(check_2024), and keep the value with the lowest WAPE per model. Never use test."*
> **Check:** the chosen values must match the `# CHECK` line.
"""),
    code('''
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


def wape_on_2024(model):
    """Learn from 2023 and return the WAPE on 2024: the number we choose a knob by."""
    model.fit(encode(learn_2023), learn_2023["units"])
    return score(check_2024["units"], model.predict(encode(check_2024)))["WAPE, %"]


candidates = {                                                     # EDIT: try other values
    "Decision tree, max_depth": {d: DecisionTreeRegressor(max_depth=d, min_samples_leaf=5, random_state=42) for d in (4, 6, 8)},
    "Random forest, max_features": {m: RandomForestRegressor(n_estimators=300, max_features=m, random_state=42, n_jobs=-1)
                                    for m in (0.33, 0.5, 1.0)},
    "LightGBM, num_leaves": {n: LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=n, min_child_samples=10,
                                              random_state=42, verbose=-1) for n in (7, 15, 31)},
}
chosen, valid_errors = {}, {}
for knob, models in candidates.items():
    errors = valid_errors[knob] = {value: wape_on_2024(model) for value, model in models.items()}
    chosen[knob] = min(errors, key=errors.get)
    print(f"{knob:28s}", "   ".join(f"{value}: {e:.1f} %" for value, e in errors.items()), f"  -> chosen: {chosen[knob]}")
# CHECK: chosen: max_depth {{nb2.chosen.tree}}, max_features {{nb2.chosen.forest}}, num_leaves {{nb2.chosen.lgbm}}
'''),
    md("""
**What to see.** The differences on 2024 are small, often below one percentage point: the knobs matter less than the inputs. One validation year is also a small sample, so do not read much into a few tenths. What matters is the habit: **every choice is made on 2024, never on 2025.**

## 5. The final test on 2025

The three models learn again, from 2023 and 2024, with the chosen knobs, and forecast every week of 2025, once.
"""),
    code('''
tree = DecisionTreeRegressor(max_depth=chosen["Decision tree, max_depth"], min_samples_leaf=5, random_state=42)
forest = RandomForestRegressor(n_estimators=300, max_features=chosen["Random forest, max_features"], random_state=42, n_jobs=-1)
boost = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=chosen["LightGBM, num_leaves"], min_child_samples=10,
                      random_state=42, verbose=-1)
for name, model in [("Decision tree", tree), ("Random forest", forest), ("LightGBM", boost)]:
    model.fit(X_train, y_train)                      # learn again, from 2023 and 2024
    test[name] = model.predict(X_test)               # forecast every week of 2025

METHODS = ["Repeat last year", "Linear regression", "Decision tree", "Random forest", "LightGBM"]
results = pd.DataFrame({m: score(test["units"], test[m]) for m in METHODS}).T
results["WAPE on 2023-24 (learned from), %"] = [np.nan, score(y_train, reg.predict(train))["WAPE, %"],
                                                *[score(y_train, m.predict(X_train))["WAPE, %"] for m in (tree, forest, boost)]]
results
# CHECK: WAPE on 2025: {{nb2.results.Repeat last year.WAPE|.1f}}, {{nb2.results.Linear regression.WAPE|.1f}}, {{nb2.results.Decision tree.WAPE|.1f}}, {{nb2.results.Random forest.WAPE|.1f}}, {{nb2.results.LightGBM.WAPE|.1f}} %
'''),
    md("""
**What to see.** On 2025, LightGBM's misses add up to {{nb2.results.LightGBM.WAPE|.1f}} % of all cases sold, the random forest's to {{nb2.results.Random forest.WAPE|.1f}} %, the regression's to {{nb2.results.Linear regression.WAPE|.1f}} % and repeating last year to {{nb2.results.Repeat last year.WAPE|.1f}} %. In our data, one tree is about as good as the regression, and many trees together are clearly better.

The last column is the error on the weeks the models learned from. The forest and LightGBM are much better there than on 2025: they know those weeks. That is why only the test year counts.

## 6. Which products gain?

The same comparison, product by product (WAPE, %):
"""),
    code('''
def wape_by_product(table, methods):
    """The WAPE of every method, product by product, with the same score function as above."""
    rows = {product: {m: score(part["units"], part[m])["WAPE, %"] for m in methods}
            for product, part in table.groupby("product")}
    return pd.DataFrame(rows).T


by_product = wape_by_product(test, METHODS)
by_product.round(1)
# CHECK: Ice tea: {{nb2.by_product.Ice tea.Linear regression|.1f}} % with the regression, {{nb2.by_product.Ice tea.LightGBM|.1f}} % with LightGBM
'''),
    md("""
**What to see.** The trees gain most where the weather and the promotions drive the sales: ice tea, radler, apple spritzer. Two products stand out. Mulled wine follows the calendar, and the models have seen only two Advent seasons: repeating last year is best in our data. Orange juice hardly reacts to the weather, yet the regression misses {{nb2.by_product.Orange juice.Linear regression|.1f}} % of its cases. Why?

The regression of all products gives **every product the same slopes**: one degree warmer adds the same number of cases of orange juice as of ice tea. A fairer test gives each product a regression of its own:
"""),
    code('''
local = []
for product, part in data.groupby("product"):                         # one regression per product
    own = smf.ols("units ~ week_of_year + temp_forecast + discount_pct + holiday_days"
                  " + last_week + avg_last_4 + same_week_last_year",
                  data=part[part["week_start"].dt.year <= 2024]).fit()
    local.append(own.predict(part[part["week_start"].dt.year == 2025]))
test["Regression per product"] = pd.concat(local)                       # lines up with the test rows by their index

side = ["Linear regression", "Regression per product", "Random forest", "LightGBM"]
per_product = wape_by_product(test, side)
per_product.loc["All products"] = [score(test["units"], test[m])["WAPE, %"] for m in side]
per_product.round(1)
# CHECK: Orange juice: {{nb2.local.Orange juice|.1f}} % with its own regression; Radler: {{nb2.local.Radler|.1f}} % against {{nb2.by_product.Radler.LightGBM|.1f}} % with LightGBM
'''),
    md("""
**What to see.** With a regression of its own, orange juice's misses add up to only {{nb2.local.Orange juice|.1f}} % of its cases sold, less than with either tree model: in this test, for this product, the straight line forecasts best (which does not prove that the business relationship is a straight line). For ice tea and radler, whose sales bend with the temperature, the trees stay clearly ahead ({{nb2.local.Ice tea|.1f}} % against {{nb2.by_product.Ice tea.Random forest|.1f}} % for ice tea; {{nb2.local.Radler|.1f}} % against {{nb2.by_product.Radler.LightGBM|.1f}} % for radler). Over all products the regressions per product reach {{nb2.local.All products|.1f}} %, LightGBM {{nb2.results.LightGBM.WAPE|.1f}} %.

A regression could also follow curves and interactions, if you write them into the formula; a tree finds them by itself. This test shows which of the tested models forecast 2025 better, not the true shape of the business relationship. These product winners describe 2025; choosing a separate model for each product from this table would need yet another untouched year to test that choice. **The practical rule:** compare a regression and a tree on a validation year, choose there, and report the chosen one on the test year, against a benchmark.

## 7. OPTIONAL: look at one product
"""),
    code('''
product = "Ice tea"            # EDIT: any of the eight products
show = ["Linear regression", "Random forest", "LightGBM"]

one = test[test["product"] == product]
assert len(one), f"No product called {product!r}; choose one of {PRODUCTS}"
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(one["week_start"], one["units"], color=COLOURS["Actual sales"], lw=2, marker="o", ms=3, label="Actual sales")
for m in show:
    ax.plot(one["week_start"], one[m], color=COLOURS[m], lw=1.8, label=m)
ax.set(title=f"{product} in 2025", ylabel="cases per week")
ax.legend(ncol=4, fontsize=9)
plt.show()
score(one["units"], one["LightGBM"]).round(1)
'''),
    md("""
**Try it.** Change `product` to Radler, Energy drink and Mulled wine. Where does LightGBM follow the actual sales, and where not? Look at the week of 30 June 2025, the hottest of all: we come back to it in notebook 3.

**Try more knobs** (optional): in section 4, add other values, for example `n_estimators` of the forest or `min_child_samples` of LightGBM, run section 4 again, and see whether the choice on 2024 changes. Leave section 5 alone: once you have seen 2025, a new choice would need another untouched year to be tested fairly.

### Your answer (not graded; it prepares your project)

1. Which method has the lowest WAPE on 2025, and how much lower is it than the regression? ___
2. Name one product where the trees gain a lot, and one where a regression is enough. Why the difference? ___
3. Why are the knobs chosen on 2024, and what would go wrong if you chose them on 2025? ___

**Next:** in notebook 3 we check a forecast that looks too good, find what a tree cannot do, and ask the model for its reasons.
"""),
]

# ============================================================================================ notebook 3
REBUILD = code('''
# Rebuild notebook 2 in one go: inputs, the time split and the models with the knobs chosen there on 2024 (a few seconds).
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

sales = sales.sort_values(["product", "week_start"]).reset_index(drop=True)
for product, part in sales.groupby("product"):                     # every week once, no gaps, before shifting
    assert (part["week_start"].diff().dropna() == pd.Timedelta(days=7)).all(), product
by_product = sales.groupby("product")["units"]
sales["last_week"] = by_product.shift(1)
sales["avg_last_4"] = by_product.transform(lambda s: s.shift(1).rolling(4).mean())
sales["same_week_last_year"] = by_product.shift(52)
sales["week_of_year"] = sales["week_start"].dt.isocalendar().week.astype(int)

FEATURES = ["product", "week_of_year", "temp_forecast", "discount_pct", "holiday_days",
            "last_week", "avg_last_4", "same_week_last_year"]
PRODUCTS = sorted(sales["product"].unique())
COLUMNS = FEATURES[1:] + ["product_" + p.replace(" ", "_") for p in PRODUCTS]


def encode(table, features=FEATURES):
    """The model inputs as numbers: the inputs, plus one 0/1 column per product."""
    X = pd.get_dummies(table[features], columns=["product"], dtype=int)
    X.columns = [c.replace(" ", "_") for c in X.columns]
    return X.reindex(columns=[c for c in COLUMNS if c in X.columns or c.startswith("product_")], fill_value=0)


@@SCORE@@


data = sales.dropna(subset=["same_week_last_year"])
train = data[data["week_start"].dt.year <= 2024]
test = data[data["week_start"].dt.year == 2025].copy()
X_train, y_train, X_test, y_test = encode(train), train["units"], encode(test), test["units"]
reg = smf.ols("units ~ C(product) + week_of_year + temp_forecast + discount_pct + holiday_days"
              " + last_week + avg_last_4 + same_week_last_year", data=train).fit()
forest = RandomForestRegressor(n_estimators=300, max_features={{nb2.chosen.forest}}, random_state=42, n_jobs=-1).fit(X_train, y_train)
boost = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves={{nb2.chosen.lgbm}}, min_child_samples=10,
                      random_state=42, verbose=-1).fit(X_train, y_train)
test["Linear regression"], test["Random forest"], test["LightGBM"] = reg.predict(test), forest.predict(X_test), boost.predict(X_test)
print("LightGBM, WAPE on 2025:", round(score(y_test, test["LightGBM"])["WAPE, %"], 1), "%")
# CHECK: LightGBM, WAPE on 2025: {{nb2.results.LightGBM.WAPE|.1f}} %
''')
REBUILD = ("code", REBUILD[1].replace("@@SCORE@@", SCORE_SRC))     # the same score function as in notebooks 1 and 2

NB3 = [
    header(3, """
**Hands-on 3.** Before you trust a forecast, check it; before you use it, understand it. You will

- check code that an AI assistant wrote, and find out why its result is too good to be true,
- see what a tree does with weeks unlike anything it has seen,
- ask the model what it relies on, how the temperature effect bends, and why one forecast is what it is,
- ask the model what-if questions.
"""),
    FIRST,
    REBUILD,
    md("""
## 1. An AI's answer to check

I asked an AI assistant: *"Here is drinks_weekly.csv with weekly sales per product. Write Python code that forecasts the weekly cases with a random forest and tells me how accurate it is."* This is its answer, unchanged. Run it.
"""),
    code('''
# ---------------- code written by an AI assistant, unchanged ----------------
from sklearn.model_selection import train_test_split

df = pd.read_csv(f"{DATA}/drinks_weekly.csv", parse_dates=["week_start"])
df = df.sort_values(["product", "week_start"])
df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
df["last_week"] = df.groupby("product")["units"].shift(1)
df = df.dropna()

X = pd.get_dummies(df.drop(columns=["units", "week_start"]), columns=["product"], dtype=int)
y = df["units"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
pred = model.predict(X_te)
print("WAPE:", round(np.abs(pred - y_te).sum() / y_te.sum() * 100, 1), "%. An excellent forecast!")
# -----------------------------------------------------------------------------
# CHECK: WAPE: {{nb3.ai.wape|.1f}} %
'''),
    md("""
**Too good to be true?** Our best honest model in notebook 2 missed {{nb2.results.LightGBM.WAPE|.1f}} % of the cases, repeating last year {{nb2.results.Repeat last year.WAPE|.1f}} %. The AI's forest claims {{nb3.ai.wape|.1f}} % with less effort. Before you celebrate, run the checks.

**Check 1: what does the model rely on?** A random forest records how much each input helped its questions.
"""),
    code('''
shares = pd.Series(model.feature_importances_, index=X.columns) * 100      # percent of all the importance
shares.sort_values(ascending=False).head(6)
# CHECK: pallets comes first, with {{nb3.ai.top_share_pct|.1f}} % of all the importance
'''),
    md("""
**Check 2: is every input known on Sunday evening?** `X` holds every column except `units` and `week_start`:

| Input | Known on Sunday evening? |
|---|---|
| `pallets` | **no**: the pallets picked for these very cases, the answer in another unit (cases ÷ cases per pallet) |
| `temp_actual` | **no**: the weather that came, known only after the week |
| `temp_forecast`, `discount_pct`, `holiday_days`, `week_of_year`, `last_week`, product | yes |

This is the overlap test from session 2: *does the input measure the outcome itself?*

**Check 3: how was it tested?** `train_test_split` picks test weeks at random, so the model learned from weeks after the ones it forecasts.

Now fix it: drop the two columns that are unknown on Sunday, then split by time.
"""),
    code('''
honest = df.drop(columns=["pallets", "temp_actual"])                          # only what we know on Sunday
X_h = pd.get_dummies(honest.drop(columns=["units", "week_start"]), columns=["product"], dtype=int)
steps = {"AI's code": round(np.abs(pred - y_te).sum() / y_te.sum() * 100, 1)}

X_tr, X_te, y_tr, y_te = train_test_split(X_h, y, test_size=0.2, random_state=42)      # still a random split
p = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_tr, y_tr).predict(X_te)
steps["without pallets and temp_actual"] = round(np.abs(p - y_te).sum() / y_te.sum() * 100, 1)

past, future = honest["week_start"].dt.year <= 2024, honest["week_start"].dt.year == 2025   # split by time
p = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_h[past], y[past]).predict(X_h[future])
steps["... and split by time"] = round(np.abs(p - y[future]).sum() / y[future].sum() * 100, 1)
pd.Series(steps, name="WAPE, %")
# CHECK: {{nb3.ai.steps.0|.1f}} %, then {{nb3.ai.steps.1|.1f}} %, then {{nb3.ai.steps.2|.1f}} %
'''),
    md("""
**What to see.** Without the answer hidden in `pallets`, the error on the same random test weeks jumps from {{nb3.ai.steps.0|.1f}} % to {{nb3.ai.steps.1|.1f}} %. Tested on 2025 after learning from the years before, it is {{nb3.ai.steps.2|.1f}} %: an honest historical test, though not a promise for next year. (This repaired version keeps the AI's own inputs and learns from 2022 to 2024, so it differs a little from notebook 2.) The code ran without a single error message: **code that runs is not code that is right.**

> **Ask an AI, the better way** · *"I forecast next week's cases per product every Sunday evening. drinks_weekly.csv has one row per product and week. Sort by product and date, then build the inputs: week of the year, temp_forecast, discount_pct, holiday_days, last week's sales via shift(1) per product, and the benchmark, sales 52 rows back per product. Do not use pallets or temp_actual: they are known only after the week. Train a random forest on 2023 and 2024, test it on 2025, and compare its WAPE with the benchmark's. Print the date range of both parts, the WAPE on the training and the test years, and a chart of actual against forecast for one product. Ask me if anything is unclear instead of guessing."*
> **Check:** the five checks from the slides: split by time, every input known on Sunday, a benchmark, the error on seen against new weeks, and a look at the forecast chart.

## 2. Weeks unlike the past

A tree answers with averages of weeks it has seen. What happens when a week looks like none of them?

**The growing product.** The energy drink sells more every year. A forest without the history inputs knows only the calendar, the weather and the promotions:
"""),
    code('''
from sklearn.base import clone

no_history = ["product", "week_of_year", "temp_forecast", "discount_pct", "holiday_days"]
forest_nh = clone(forest)                                        # the same settings, only the inputs differ
forest_nh.fit(encode(train, no_history), y_train)
test["Forest without history"] = forest_nh.predict(encode(test, no_history))

energy = test[test["product"] == "Energy drink"]
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(energy["week_start"], energy["units"], color=COLOURS["Actual sales"], lw=2, marker="o", ms=3, label="Actual sales")
ax.plot(energy["week_start"], energy["Random forest"], color=COLOURS["Random forest"], lw=1.8, label="Random forest")
ax.plot(energy["week_start"], energy["Forest without history"], color=COLOURS["Random forest"], lw=1.8, ls="--",
        label="Random forest without history")
ax.set(title="Energy drink in 2025", ylabel="cases per week")
ax.legend(fontsize=9)
plt.show()
print("2025 average:", round(energy["units"].mean()), "| forest:", round(energy["Random forest"].mean()),
      "| without history:", round(energy["Forest without history"].mean()),
      "| highest week in 2023-24:", train.loc[train["product"] == "Energy drink", "units"].max())
# CHECK: 2025 average: {{nb3.energy.actual}} | forest: {{nb3.energy.forest}} | without history: {{nb3.energy.no_history}}
'''),
    md("""
**What to see.** Without the history inputs the forest stays near the level of 2023 and 2024: a tree forecasts averages of training weeks (in a model for all products, those can include weeks of other products), so it does not continue a trend by itself. With last week and the last four weeks as inputs it follows the growth most of the way; the inputs carry the new level. **Give a tree the recent level, or let it forecast the change, when a series grows.**

**The record heat.** The week of 30 June 2025 had a forecast of {{nb3.heat.temp|.1f}} °C, hotter than any week the models learned from (at most {{nb3.heat.train_max|.1f}} °C):
"""),
    code('''
heat = test[test["week_start"].isin(pd.to_datetime(["2025-06-23", "2025-06-30", "2025-07-07"]))
            & test["product"].isin(["Ice tea", "Mineral water"])]
print("Hottest forecast in 2023-24:", train["temp_forecast"].max(), "°C")
heat[["week_start", "product", "temp_forecast", "discount_pct", "units", "Linear regression", "LightGBM"]].round({"Linear regression": 0, "LightGBM": 0})
# CHECK: ice tea in the week of 30 June: {{nb3.heat.ice_actual}} cases sold, LightGBM {{nb3.heat.ice_lgbm}}
'''),
    md("""
**What to see.** In the record week, with a promotion on top, ice tea sold {{nb3.heat.ice_actual}} cases; LightGBM forecast {{nb3.heat.ice_lgbm}}. LightGBM adds up the corrections of many trees, so it can step a little outside the sales it has seen; but once the temperature is beyond its hottest learned split, more heat no longer changes its forecast. No week in the training data looked like this one. Flag weeks unlike the past (record weather, a first-time promotion, a new product) and plan them with judgement on top of the model.

## 3. What the model relies on

**Permutation importance:** shuffle one input in the 2025 weeks, so its values no longer belong to their weeks, and see how much the WAPE rises. A big rise means the model relies on that input for its 2025 forecasts. Each input is shuffled **within the same product**, so radler never gets the sales scale of mineral water; the product itself is swapped between rows. Shuffling can still create unusual weeks (summer weather with a winter history), and inputs that carry the same information can stand in for each other, so a small value does not prove an input useless.
"""),
    code('''
def importance(model, table, repeats=3):
    """How much the WAPE on `table` rises when one input is shuffled (average of a few shuffles)."""
    base = score(table["units"], model.predict(encode(table)))["WAPE, %"]
    rng = np.random.default_rng(0)
    rise = {}
    for feature in FEATURES:
        rises = []
        for _ in range(repeats):
            shuffled = table.copy()
            if feature == "product":                     # swap the product between rows
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            else:                                        # shuffle within each product
                for _, idx in shuffled.groupby("product").groups.items():
                    shuffled.loc[idx, feature] = rng.permutation(shuffled.loc[idx, feature].to_numpy())
            rises.append(score(table["units"], model.predict(encode(shuffled)))["WAPE, %"] - base)
        rise[feature] = np.mean(rises)
    return pd.Series(rise).sort_values()


relies = importance(boost, test)
relies.plot.barh(color="#0B5CAD", figsize=(7, 3.6), title="LightGBM: rise in WAPE when an input is shuffled")
plt.xlabel("percentage points")
plt.show()
relies.sort_values(ascending=False).round(1)
# CHECK: the three largest: {{nb3.importance.top3}}
'''),
    md("""
**What to see.** The model relies most on the planned discount and the temperature forecast, then on which product it is and on last year's level. These are the habits of this fitted model in its 2025 forecasts, not proof of what causes demand. If an input you know to be irrelevant ranks high, look for a leak (as with `pallets` above).

## 4. OPTIONAL: how the forecast changes with the temperature

Take one real week per product, change only the temperature forecast, and ask the model again for every temperature from -5 to 35 °C. These curves describe the model for these weeks, not what temperature causes; some combinations (frost with an August sales history) never happen. The regression of session 2 answers with one straight line and the same slope for every product.
"""),
    code('''
weeks = {"Ice tea": "2025-08-25", "Radler": "2025-08-25", "Mineral water": "2025-08-25"}   # EDIT: product and week
temps = np.arange(-5, 36)
PRODUCT_COLOURS = {"Mineral water": "#0B5CAD", "Ice tea": "#E66A1F", "Mulled wine": "#1B9E5A", "Radler": "#D04F95"}

fig, ax = plt.subplots(figsize=(8, 4))
checks = []
for product, week in weeks.items():
    row = test[(test["product"] == product) & (test["week_start"] == week)]
    assert len(row) == 1, f"Choose a product and a Monday of 2025 that exist: {product}, {week}"
    what_if = pd.concat([row] * len(temps), ignore_index=True)
    what_if["temp_forecast"] = temps
    forecast = boost.predict(encode(what_if))
    ax.plot(temps, forecast, color=PRODUCT_COLOURS.get(product, "#595959"), lw=2.5, label=f"{product}, week of {week}")
    checks.append({"product": product, "at 10 °C": forecast[temps == 10][0], "at 30 °C": forecast[temps == 30][0]})
ax.set(xlabel="temperature forecast, °C", ylabel="forecast, cases", title="LightGBM: the same week at every temperature")
ax.legend(fontsize=9)
plt.show()
print(pd.DataFrame(checks).round(0).to_string(index=False))
print("The regression: one degree warmer adds", round(reg.params["temp_forecast"], 1), "cases, for every product")
# CHECK: the regression: one degree warmer adds {{nb3.curves.reg_slope|.1f}} cases, for every product
'''),
    md("""
**What to see.** Read the printed forecasts at 10 and 30 °C: each product has its own curve, flat in the cold and climbing in summer temperatures. The regression gives one number per input, {{nb3.curves.reg_slope|.1f}} cases per degree, the same for every product and every temperature. **A tree model gives each product its own curve.** Beyond about {{nb3.heat.train_max|.0f}} °C every curve goes flat: the model has never seen hotter weeks. The curves need not rise steadily; small dips are the model's steps, not a business effect.

Mulled wine is missing on purpose. The models have seen only two Advent seasons, about a dozen weeks: too few to learn how the cold moves its sales. Few examples, no reliable pattern.

## 5. One forecast explained

LightGBM can split one forecast into a **reference**, the model's average forecast over the training weeks, plus one share per input, in cases. These shares are called **SHAP values**; reference plus shares give exactly the forecast. They describe how the model built this number, not what would happen if you changed an input (that is the what-if question of section 6).
"""),
    code('''
product, week = "Radler", "2025-06-23"                      # EDIT: any product and any Monday of 2025
row = test[(test["product"] == product) & (test["week_start"] == week)]
assert len(row) == 1, "Choose a product and a Monday of 2025 that exist."

parts = pd.Series(boost.predict(encode(row), pred_contrib=True)[0], index=COLUMNS + ["reference: average forecast"])
assert np.isclose(parts.sum(), boost.predict(encode(row))[0])          # reference plus shares = the forecast
product_columns = [c for c in COLUMNS if c.startswith("product_")]
parts["product"] = parts[product_columns].sum()              # the eight 0/1 columns as one line
parts = parts.drop(product_columns)
values = row[FEATURES].iloc[0]
table = pd.DataFrame({"value this week": [values.get(i, "") for i in parts.index], "share, cases": parts.round(0)})
print(f"{product}, week of {week}: forecast {parts.sum():,.0f} cases, actual {row['units'].iloc[0]:,} cases")
table.sort_values("share, cases", key=abs, ascending=False)
# CHECK: Radler, week of 2025-06-23: forecast {{nb3.explain.forecast|,.0f}} cases, actual {{nb3.explain.actual|,.0f}} cases
'''),
    md("""
**How to read it.** The reference is the model's average forecast. Each line moves this week's forecast up or down from there; the shares add up to the forecast. In a meeting: *"Our model forecasts about {{nb3.explain.forecast|,.0f}} cases of Radler. Compared with its average week, it attributes about {{nb3.explain.discount_part|,.0f}} cases to the planned {{nb3.explain.discount}} % promotion and about {{nb3.explain.temp_part|,.0f}} to the {{nb3.explain.temp|.0f}} °C forecast."* Change the week and the product, and explain another forecast.

## 6. OPTIONAL: what if?

The model answers questions a planner asks every week: what if we skip the promotion, what if the heat breaks and the week stays cool? These are the model's forecasts under other inputs, with everything else kept; they are not proof of cause. Note that "without the promotion" changes the forecast by more than the promotion's SHAP share above: the two answer different questions.
"""),
    code('''
plans = {"as planned": row.copy(), "without the promotion": row.copy(), "a cool week, 18 °C": row.copy()}
plans["without the promotion"]["discount_pct"] = 0                                    # EDIT: 0, 10, 20 or 30
plans["a cool week, 18 °C"]["temp_forecast"] = 18                                     # EDIT: any temperature
for name, plan in plans.items():
    print(f"{name:27s}", round(boost.predict(encode(plan))[0]), "cases")
# CHECK: as planned {{nb3.what_if.as_planned}}, without the promotion {{nb3.what_if.no_promo}}, a cool week {{nb3.what_if.cool}}
'''),
    md("""
### Your answer (not graded; it prepares your project)

1. Name the two inputs in the AI's code that are unknown on Sunday evening, and say how you found the first one. ___
2. Why does a forest without the history inputs fail on the energy drink? ___
3. Write the meeting sentence for one forecast of your choice from section 5: *"Our model forecasts ___ cases of ___ for the week of ___; compared with its average week, it attributes the most to ___."*
"""),
]

# ============================================================================================ notebook 4
NB4 = [
    md(f"""
# {NB_TITLES[4]}

<a href="{colab_url(4)}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

*{COURSE}*

Use this notebook if you choose **tree-based models** for your project. It follows the steps of the session, with every check you need for your report.

**Which module fits your question?** Choose Taha's (regression) when your main question is about relationships, effect sizes and their uncertainty. Choose Mehran's (clustering) when you look for groups of similar objects and have no target to predict. Choose this one when you want to forecast a number for future periods and can test the forecasts against later observations.

**How to use it**

1. Run it once **unchanged**: it works with the course data (`drinks_weekly.csv`) and shows what every step produces.
2. Then change the `# EDIT` lines to your own table, **one section at a time**, and run the notebook again from the top. Every check stops with a message that says what to fix.
3. Write your decisions in the cells marked **Your decision**. They become the backbone of your report.

**Your table** needs one row per series and regular period (one product per week, one warehouse per day, one region per month), a date column, a number to forecast, and inputs. With one single series, set `group = None`. The template forecasts the next period. A survey, an irregular list of shipments or a table without dates needs a different preparation and test: bring it to the consultation.

`# CHECK` lines give the result for the unchanged course data; with your data the numbers change. They can also differ a little from notebook 2: this template orders the inputs differently (a random forest then draws other inputs at its questions, even with the same seed) and sets forecasts below zero to zero.

| Section | Report criterion (weight) |
|---|---|
| 1 | Question and SCM relevance (15 %) |
| 2 and 3 | Data preparation and features (20 %) |
| 4 and 5 | Method and implementation (20 %) |
| 5 and 6 | Checks and experiments (25 %): your experiment on validation, the test once, the five checks |
| 7 and 8 | Interpretation and presentation (20 %) |
"""),
    md("""
## 1 · Your question

> **Your decision (question):**
>
> - Which decision will use the forecast, and who makes it? *(course data: how many cases each product will sell next week, for the orders to the bottlers and the warehouse plan; stock and lead times then set the order itself)*
> - What exactly do you forecast, in which unit? *(cases per product per week)*
> - What is one row? *(one product in one week)*
> - When do you make the forecast, and how far ahead? *(Sunday evening, for the coming week)*
> - Why a tree-based model and not a regression or clustering? *(the effect of the weather and the promotions depends on the situation, and we want a forecast)*
"""),
    md("""
## 2 · Load and check your table

**Edit** the file, its format, the names of the date, target and group columns, and how often a period comes (`frequency`). Missing targets must be resolved in the source data: never fill them from later periods, and remember that dropping a period changes what "the period before" means.
"""),
    code(f'''
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# EDIT: your file and its format
file = "{RAW}/drinks_weekly.csv"     # a link, or a path such as "data/my_table.csv" after you uploaded or mounted it
separator = ","                      # "," or ";"
decimal = "."                        # "." for 1.5, "," for 1,5
date_column = "week_start"           # the column with the date of the period
date_format = None                   # None for dates like 2025-01-31; "%d.%m.%Y" for 31.01.2025
target = "units"                     # the number you forecast
group = "product"                    # the series a row belongs to (product, store, route), or None for one series
frequency = "W-MON"                  # "W-MON" weekly on Mondays, "MS" monthly (first day), "D" daily

table = pd.read_csv(file, sep=separator, decimal=decimal)
needed = [date_column, target] + ([group] if group else [])
assert len(table), "The file has no rows."
assert len(set(needed)) == len(needed), "Date, target and group must be three different columns."
missing_columns = [c for c in needed if c not in table.columns]
assert not missing_columns, f"Not found: {{missing_columns}}. Check the names and the separator. The file has: {{list(table.columns)}}"
table[date_column] = pd.to_datetime(table[date_column], format=date_format, errors="coerce")
table[target] = pd.to_numeric(table[target], errors="coerce")
assert table[needed].notna().all().all(), "Some dates, targets or series labels are missing or unreadable: check date_format and decimal."
assert np.isfinite(table[target].to_numpy(dtype=float)).all(), "The target must hold finite numbers."
keys = ([group] if group else []) + [date_column]
assert not table.duplicated(keys).any(), "A period appears twice for a series: aggregate to one row per series and period first."
table = table.sort_values(keys).reset_index(drop=True)
for key, part in (table.groupby(group) if group else [("your series", table)]):
    expected = pd.date_range(part[date_column].min(), part[date_column].max(), freq=frequency)
    assert len(expected) == len(part) and (expected == pd.DatetimeIndex(part[date_column])).all(), \\
        f"{{key}}: periods are missing or do not match frequency={{frequency!r}}. shift() counts rows, so every period must be there once."
print("rows and columns:", table.shape)
print("from", table[date_column].min().date(), "to", table[date_column].max().date(), "|", table[group].nunique() if group else 1, "series")
table.head()
# CHECK for the course data: ({{{{data.rows}}}}, {{{{data.columns}}}}), from {{{{data.first_week}}}} to {{{{data.last_week}}}}, {{{{data.products}}}} series
'''),
    md("""
## 3 · Inputs known at decision time

List your inputs in two groups. **Known inputs** are known for the period you forecast (a planned promotion, a holiday, a weather forecast). **History inputs** are built from the target itself, always from earlier periods only. Then fill in the table in **Your decision** for every column of your file, including the ones you leave out.
"""),
    code('''
known_inputs = ["temp_forecast", "discount_pct", "holiday_days"]   # EDIT: known when you forecast (numbers or text)
not_known = ["pallets", "temp_actual"]     # EDIT: columns you must NOT use: unknown at decision time (say why in the report)
seasonal_lag = 52            # EDIT: about one year back: 52 (weekly, 364 days), 12 (monthly), 364 (daily)
lags = [1, seasonal_lag]     # EDIT: history inputs, in periods back; 1 = the period just before
window = 4                   # EDIT: the average of the last 4 periods
extra_categories = []        # EDIT: number columns that are really labels (a store number, a route code)

assert lags and all(type(k) is int and k > 0 for k in lags) and len(set(lags)) == len(lags), "lags: different whole numbers above 0."
assert type(window) is int and window > 0, "window: a whole number above 0."
absent = [c for c in known_inputs + not_known if c not in table.columns]
assert not absent, f"Not in your table: {absent}"
history = [f"lag_{k}" for k in lags] + [f"avg_last_{window}"]
clash = [c for c in history + ["season"] if c in table.columns]
assert not clash, f"Rename these columns of your file first, the template creates them: {clash}"
shortest = table.groupby(group).size().min() if group else len(table)
assert shortest > max(lags + [window]) + 10, "A series is too short for these lags: use shorter lags or a longer history."

past = table.groupby(group)[target] if group else table[target]
for k in lags:
    table[f"lag_{k}"] = past.shift(k)                       # k periods back, within each series
table[f"avg_last_{window}"] = (past.transform(lambda s: s.shift(1).rolling(window).mean()) if group
                               else table[target].shift(1).rolling(window).mean())


def season_of(dates):
    """The calendar input: the week of the year (weekly data), the month (monthly) or the weekday (daily)."""
    if frequency.startswith("W"):
        return dates.dt.isocalendar().week.astype(int)
    if frequency in ("MS", "ME", "M"):
        return dates.dt.month
    return dates.dt.dayofweek


table["season"] = season_of(table[date_column])
inputs = known_inputs + history + ["season"] + ([group] if group else [])
assert len(set(inputs)) == len(inputs), "An input appears twice."
assert not set(inputs) & set(not_known + [target, date_column]), "An input is the target, the date, or unknown at decision time."
model_table = table.dropna(subset=history).copy()
print(len(model_table), "rows with a complete history;", len(table) - len(model_table), "early rows only feed the history")
model_table[[date_column] + ([group] if group else []) + [target] + history].head()
'''),
    md("""
> **Your decision (inputs):** for every column, one line:
>
> | Column | Use it? | Known at decision time? | Why |
> |---|---|---|---|
> | *temp_forecast* | *yes* | *yes, the forecast on Sunday* | *drinks follow the weather* |
> | *pallets* | *no* | *no, only after the week* | *the target in another unit* |
"""),
    md("""
## 4 · Three periods: learn, choose, test

The models learn from the first period, you **choose** on the validation period, and the test period is used **once**, at the end, after every choice is written down. Choose periods long enough to contain the situations that matter (for seasonal data, a whole year each).

**What the test means:** each test row is a separate forecast for the next period, made with the plans known then and the sales of the periods already finished (assumed final by then). Later test forecasts can therefore use earlier test sales. This does not test forecasting the whole test period at once.
"""),
    code('''
validation_start = "2024-01-01"    # EDIT: the first date of the validation period (to choose)
test_start = "2025-01-01"          # EDIT: the first date of the test period (used once, at the end)
fill_missing = False               # EDIT: True fills missing number inputs with their median in the first period

dates = model_table[date_column]
is_fit = dates < validation_start
is_valid = (dates >= validation_start) & (dates < test_start)
is_test = dates >= test_start
is_train = ~is_test                                 # learn and validation together: what the final models learn from
for name, mask in [("learn", is_fit), ("validation", is_valid), ("test", is_test)]:
    assert mask.sum() >= 10, f"The {name} period has only {int(mask.sum())} rows: move validation_start and test_start."

text_inputs = [c for c in inputs if c == group or c in extra_categories or not pd.api.types.is_numeric_dtype(model_table[c])]
number_inputs = [c for c in inputs if c not in text_inputs]
gaps = model_table[inputs].isna().sum()
if gaps.any():
    assert fill_missing, f"Missing values in the inputs: {gaps[gaps > 0].to_dict()}. Fix them in the data, or set fill_missing = True."
    for c in gaps[gaps > 0].index:
        assert c in number_inputs, f"{c} is a text input: fill its missing values in the data."
        model_table[c] = model_table[c].fillna(model_table.loc[is_fit, c].median())    # learned from the first period only
categories = {c: sorted(model_table.loc[is_fit, c].astype(str).unique()) for c in text_inputs}   # learned from the first period


def check_inputs(rows):
    """Stop with a message if an input is missing, not a number, infinite, or a label the first period never had."""
    left = rows[inputs].isna().sum()
    assert not left.any(), f"Missing values left in the inputs: {left[left > 0].to_dict()}"
    assert all(pd.api.types.is_numeric_dtype(rows[c]) for c in number_inputs), "A measurement is not a number: check decimal."
    assert np.isfinite(rows[number_inputs].to_numpy(dtype=float)).all(), "A number input holds an infinite value."
    for c in text_inputs:
        unseen = sorted(set(rows[c].astype(str)) - set(categories[c]))
        assert not unseen, (f"{c}: labels absent from the first period, for example {unseen[:3]}. "
                            "Decide in the report how to handle new labels; do not just drop rows.")


def as_numbers(rows, columns=None):
    """The inputs as numbers: each text input becomes one 0/1 column per label of the first period."""
    columns = inputs if columns is None else columns
    X = rows[columns].copy()
    for c in columns:
        if c in text_inputs:
            X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    X = pd.get_dummies(X, columns=[c for c in columns if c in text_inputs], dtype=int)
    X.columns = [re.sub(r"[^0-9A-Za-z_]", "_", str(c)) for c in X.columns]
    assert X.columns.is_unique, "Two inputs got the same name after cleaning: rename them in your file."
    return X


check_inputs(model_table)
if gaps.any():
    print("Filled with the median of the first period (say so in the report):", gaps[gaps > 0].to_dict())
X_all, y_all = as_numbers(model_table), model_table[target]
for name, mask in [("learn", is_fit), ("validation", is_valid), ("test", is_test)]:
    print(f"{name:11s}", int(mask.sum()), "rows,", dates[mask].min().date(), "to", dates[mask].max().date())
print("number inputs:", ", ".join(number_inputs), "| text inputs:", ", ".join(text_inputs) or "none")
# READ the line above: a measurement listed as a text input was misread (a decimal comma?): fix the format in section 2.
# CHECK for the course data: learn {{nb4.fit_rows}} rows from {{nb4.fit_from}}; validation {{nb4.valid_rows}} rows from {{nb4.valid_from}}; test {{nb4.test_rows}} rows from {{nb4.test_from}}
'''),
    md("""
## 5 · Choose on the validation period

Four methods: the **benchmark** repeats the value from `benchmark_lag` periods earlier; the **regression** of session 2; a **random forest** and **LightGBM**, with the knobs chosen in notebook 2 as starting values. Each learns from the first period and is scored on the validation period; the best one is your choice. Then your own **experiment**, also on the validation period. The test period stays closed until section 6.
"""),
    code('''
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


def score(actual, forecast):
    """MAE: the average miss. WAPE: all misses as a percentage of the total. Bias: forecast minus actual."""
    if isinstance(actual, pd.Series) and isinstance(forecast, pd.Series):
        assert actual.index.equals(forecast.index), "Actuals and forecasts must belong to the same rows."
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    assert a.ndim == 1 and a.size and a.shape == f.shape, "One forecast for every row, in one column."
    assert np.isfinite(a).all() and np.isfinite(f).all(), "Every actual value needs one finite forecast."
    miss, total = f - a, np.abs(a).sum()
    return pd.Series({"MAE": np.abs(miss).mean(), "WAPE, %": np.abs(miss).sum() / total * 100 if total else np.nan, "Bias": miss.mean()})


benchmark_lag = seasonal_lag   # EDIT: the period the benchmark repeats (one of the lags)
assert benchmark_lag in lags, "benchmark_lag must be one of the lags of section 3."
clip_at_zero = True            # EDIT: sales cannot be negative; False if your target can be
models = {"Random forest": RandomForestRegressor(n_estimators=300, max_features={{nb2.chosen.forest}}, random_state=42, n_jobs=-1),
          "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves={{nb2.chosen.lgbm}}, min_child_samples=10,
                                    random_state=42, verbose=-1)}
q = lambda c: f"Q({c!r})"                                     # quotes any column name for the formula
formula = q(target) + " ~ " + " + ".join(f"C({q(c)})" if c in text_inputs else q(c) for c in inputs)


def policy(prediction):
    """The same rule for every forecast: below zero becomes zero when clip_at_zero is True."""
    prediction = np.asarray(prediction, dtype=float)
    return np.clip(prediction, 0, None) if clip_at_zero else prediction


def forecast_with(method, learn_mask, rows):
    """Learn `method` from the rows of learn_mask and forecast `rows`."""
    if method == "Benchmark":
        f = rows[f"lag_{benchmark_lag}"].to_numpy(float)
    elif method == "Linear regression":
        as_text = {c: object for c in text_inputs}
        f = smf.ols(formula, data=model_table[learn_mask].astype(as_text)).fit().predict(rows.astype(as_text)).to_numpy(float)
    else:
        f = clone(models[method]).fit(X_all[learn_mask], y_all[learn_mask]).predict(as_numbers(rows))
    return policy(f)


METHODS = ["Benchmark", "Linear regression", "Random forest", "LightGBM"]
validation = pd.DataFrame({m: score(y_all[is_valid], forecast_with(m, is_fit, model_table[is_valid])) for m in METHODS}).T
assert validation["WAPE, %"].notna().all(), "WAPE is undefined (the actual values add up to zero): choose by MAE instead."
chosen_method = validation["WAPE, %"].idxmin()
print("Chosen on the validation period:", chosen_method)
validation.round(1)
# CHECK for the course data: chosen {{nb4.chosen}}; validation WAPE {{nb4.valid_wape_chosen|.1f}} %
'''),
    code('''
# EDIT: your what-if experiment, judged on the validation period. Example: LightGBM without the temperature forecast.
experiment_inputs = [c for c in inputs if c != "temp_forecast"]
X_exp = as_numbers(model_table, experiment_inputs)
exp = clone(models["LightGBM"]).fit(X_exp[is_fit], y_all[is_fit])
print("LightGBM on the validation period, all inputs:     ", round(validation.loc["LightGBM", "WAPE, %"], 1), "%")
print("LightGBM on the validation period, the experiment: ", round(score(y_all[is_valid], policy(exp.predict(X_exp[is_valid])))["WAPE, %"], 1), "%")
# CHECK for the course data: {{nb4.valid_lgbm|.1f}} % with all inputs, {{nb4.experiment_wape|.1f}} % without the temperature forecast
'''),
    md("""
> **Your decision (method), written down before you open the test:** Which methods did you compare, with which settings? Which one did the validation period choose? What did your experiment change on the validation period? If you adopt a change (an input, a knob), make it in section 3 or 5 and run again from there, so `chosen_method` and the models use it. Changes made after seeing the test results would need another untouched period to be tested fairly.

## 6 · The test, once, and the checks

All four methods learn again from both periods and forecast the test period. Then the five checks from the session.
"""),
    code('''
forecasts = pd.DataFrame({m: forecast_with(m, is_train, model_table[is_test]) for m in METHODS}, index=model_table.index[is_test])
results = pd.DataFrame({m: score(y_all[is_test], forecasts[m]) for m in METHODS}).T
print("The test period, used once. Your choice:", chosen_method)
results.round(1)
# CHECK for the course data: test WAPE: benchmark {{nb4.test_wape.Benchmark|.1f}} %, regression {{nb4.test_wape.Linear regression|.1f}} %, forest {{nb4.test_wape.Random forest|.1f}} %, LightGBM {{nb4.test_wape.LightGBM|.1f}} %
'''),
    md("""
On the course data the validation period chooses the **{{nb4.chosen}}**. Compare its test result with the others: when methods are close on the validation period, their order can change on the test period. Report what you chose on the validation period and why, not the method with the best test number.
"""),
    code('''
seen = score(y_all[is_train], forecast_with(chosen_method, is_train, model_table[is_train]))["WAPE, %"]
print("1. Split by time:          learn to", dates[is_fit].max().date(), "| validation to", dates[is_valid].max().date(),
      "| test from", dates[is_test].min().date())
print("2. Inputs known in time:   ", ", ".join(inputs), "| left out:", ", ".join(not_known) or "none")
print("3. Against the benchmark:  ", chosen_method, round(results.loc[chosen_method, "WAPE, %"], 1), "% against",
      round(results.loc["Benchmark", "WAPE, %"], 1), "% on the test period")
print("4. Seen against new data:  ", chosen_method, round(seen, 1), "% on the rows it learned from,",
      round(results.loc[chosen_method, "WAPE, %"], 1), "% on the test period")
print("5. Look at the chart below: below zero, ceilings, copies of the last period, periods unlike the past")
if group:                    # the WAPE of the chosen method for every series (undefined where the series sold nothing)
    tested = model_table[is_test]
    per_series = {key: score(part[target], forecasts.loc[part.index, chosen_method])["WAPE, %"] for key, part in tested.groupby(group)}
    print(pd.Series(per_series).sort_values().round(1).to_string())
'''),
    code('''
show_group = model_table.loc[is_test, group].iloc[0] if group else None    # EDIT: one series to look at
part = model_table[is_test] if not group else model_table[is_test & (model_table[group] == show_group)]
assert len(part), f"No test rows for {show_group!r}."
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(part[date_column], part[target], color="#262626", lw=2, marker="o", ms=3, label="Actual")
colours = {"Benchmark": "#E66A1F", "Linear regression": "#1B9E5A", "Random forest": "#D04F95", "LightGBM": "#0B5CAD"}
for m in dict.fromkeys([chosen_method, "Benchmark", "Linear regression"]):
    ax.plot(part[date_column], forecasts.loc[part.index, m], color=colours[m], lw=1.8, label=m)
ax.set(title=f"Test period: {show_group if group else target}", ylabel=target)
ax.legend()
plt.show()
'''),
    md("""
> **Your decision (checks):** What did each of the five checks show? What does your experiment of section 5 tell you about your data?

## 7 · OPTIONAL: explain the LightGBM model

What LightGBM relies on (permutation importance on the test period: other inputs shuffled within each series, the series label swapped between rows; shuffling can create unusual combinations, and related inputs can share information), and one forecast split into a reference plus one share per input (SHAP values: they describe the raw model forecast, not the effect of changing an input). If you chose another method, explain that one in your report instead.
"""),
    code('''
final_lgbm = clone(models["LightGBM"]).fit(X_all[is_train], y_all[is_train])
X_test = X_all[is_test]
feature_columns = {c: as_numbers(model_table.iloc[:5], [c]).columns.tolist() for c in inputs}   # the 0/1 columns of each input
rng = np.random.default_rng(0)
base = score(y_all[is_test], policy(final_lgbm.predict(X_test)))["WAPE, %"]
series_of = model_table.loc[is_test, group].astype(str).to_numpy() if group else np.zeros(int(is_test.sum()))
rise = {}
for c in inputs:
    rises = []
    for _ in range(3):
        order = np.arange(len(X_test))
        if c == group:                                   # swap the series between rows
            order = rng.permutation(len(X_test))
        else:                                            # shuffle within each series
            for g in np.unique(series_of):
                where = np.flatnonzero(series_of == g)
                order[where] = rng.permutation(where)
        shuffled = X_test.copy()
        shuffled[feature_columns[c]] = X_test[feature_columns[c]].to_numpy()[order]
        rises.append(score(y_all[is_test], policy(final_lgbm.predict(shuffled)))["WAPE, %"] - base)
    rise[c] = np.mean(rises)
pd.Series(rise).sort_values().plot.barh(color="#0B5CAD", figsize=(7, 3.4), title="Rise in WAPE when an input is shuffled")
plt.xlabel("percentage points")
plt.show()

i = 0                        # EDIT: which test row to explain (0 = the first)
contrib = pd.Series(final_lgbm.predict(X_test.iloc[[i]], pred_contrib=True)[0], index=list(X_test.columns) + ["reference"])
shares = pd.Series({c: contrib[feature_columns[c]].sum() for c in inputs})
raw = final_lgbm.predict(X_test.iloc[[i]])[0]
assert np.isclose(contrib["reference"] + shares.sum(), raw)      # reference plus shares = the raw forecast
print("Raw forecast:", round(raw, 1), "| after the zero rule:", round(policy([raw])[0], 1),
      "| reference (average forecast):", round(contrib["reference"], 1), "| actual:", y_all[is_test].iloc[i])
shares.sort_values(key=abs, ascending=False).round(1)
'''),
    md("""
> **Your decision (interpretation):** Which inputs does the model rely on, and does that make business sense? Explain one forecast in one sentence a manager understands. Where should nobody trust the model (periods unlike the past, new series, small series)?

## 8 · Your forecast and the decision

The chosen method learns from **all** your data and forecasts the **next period**. The history inputs of the next period come from your data; the known inputs (your plan, your weather forecast) you fill in yourself. The plan below gives every series the same values; if plans differ by series, join a plan table by series and date with `validate="one_to_one"` and check it is complete before the forecast.
"""),
    code('''
step = pd.tseries.frequencies.to_offset(frequency)
rows = []
for key, part in (table.groupby(group) if group else [(None, table)]):
    values = part[target].to_numpy()                       # the series up to the last known period
    new = {date_column: part[date_column].max() + step}
    if group:
        new[group] = key
    for k in lags:
        new[f"lag_{k}"] = values[-k]                       # k periods before the next one
    new[f"avg_last_{window}"] = values[-window:].mean()
    rows.append(new)
next_period = pd.DataFrame(rows)
next_period["season"] = season_of(next_period[date_column])
assert next_period[date_column].nunique() == 1, "The series end on different dates: choose one common forecast date."

plan = {"temp_forecast": 2.0, "discount_pct": 0, "holiday_days": 1}    # EDIT: one value for each known input
assert set(plan) == set(known_inputs), f"Give exactly one value for each known input: {known_inputs}"
for c, value in plan.items():
    next_period[c] = value
check_inputs(next_period)
next_period["forecast"] = forecast_with(chosen_method, np.ones(len(model_table), dtype=bool), next_period)
assert np.isfinite(next_period["forecast"]).all(), "Every series needs one finite forecast."
print("Forecast with", chosen_method, "for the period starting", next_period[date_column].iloc[0].date())
next_period[([group] if group else []) + [date_column, "forecast"]].round({"forecast": 1})
# CHECK for the course data: {{nb4.chosen}}, the week of {{nb4.next_week}}; Mineral water {{nb4.next_water|,.1f}} cases
'''),
    md("""
> **Your decision (the answer to your question):** What do you recommend, based on the forecast? How large were the errors on the test period (MAE, WAPE), and were the forecasts more often too high or too low? What does that mean for the decision? (Neither number sets a safety stock by itself.) What would you do next with more time or data?

**Document your use of AI** in the report: where you used it (for example "to adapt section 2 to my file") and how you checked the result. Code that runs is not code that is right.
"""),
]

NOTEBOOKS = {1: NB1, 2: NB2, 3: NB3, 4: NB4}

# ----------------------------------------------------------------------------- filling the placeholders
_PH = re.compile(r"\{\{([^{}|]+)(?:\|([^{}]+))?\}\}")


def lookup(results, path):
    node = results
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def fill(text, results):
    def rep(m):
        value = lookup(results, m.group(1).strip())
        spec = m.group(2)
        return format(value, spec) if spec else str(value)
    return _PH.sub(rep, text)


def code_cells(n):
    """The code of notebook n, cell by cell, as it is executed (placeholders sit only in comments)."""
    return [src for kind, src in NOTEBOOKS[n] if kind == "code"]


def build(results):
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for n, cells in NOTEBOOKS.items():
        nb = nbformat.v4.new_notebook()
        nb.metadata = {"colab": {"name": NB_NAMES[n], "provenance": [], "toc_visible": True},
                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}
        for kind, src in cells:
            text = fill(src, results)
            nb.cells.append(nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text))
        left = [m.group(0) for c in nb.cells for m in _PH.finditer(c.source)]
        if left:
            sys.exit(f"Unfilled placeholders in notebook {n}: {left[:5]}")
        nbformat.write(nb, out / NB_NAMES[n])
        print(f"notebooks/{NB_NAMES[n]}: {len(nb.cells)} cells")


if __name__ == "__main__":
    results = json.loads((ROOT / "evidence" / "results.json").read_text(encoding="utf-8"))
    build(results)

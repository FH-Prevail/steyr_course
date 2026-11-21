# -*- coding: utf-8 -*-
"""
build_docs.py - writes the two student-facing READMEs (README.md and data/README.md) from
evidence/results.json, so every number in them is the number the notebooks print.

    python3 scripts/build_docs.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_notebooks import REPO, NB_NAMES, NB_TITLES, ON_COURSE_PAGE, colab_url  # noqa: E402

R = json.loads((ROOT / "evidence" / "results.json").read_text(encoding="utf-8"))
D, N2 = R["data"], R["nb2"]
RES = N2["results"]
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
PDF = "slides/S4_Forecasting_With_Trees.pdf"

WHAT = {1: "Hands-on 1: the straight line from session 2 against your first decision tree, for ice tea; how deep a tree should grow",
        2: "Hands-on 2: a random forest and LightGBM for all eight products, tested on 2025 against a benchmark and the regression",
        3: "Hands-on 3: check code an AI wrote, see what a tree cannot do, and explain a forecast",
        4: "Your project: the same steps for your own table, with every check, mapped to the report criteria"}

TEMPLATE = (f"start from [notebook 4, the project template](notebooks/{NB_NAMES[4]})." if 4 in ON_COURSE_PAGE else
            "you get notebook 4, the project template, from me: write to me and I send it to you.")
nb_rows = "\n".join(f"| [{NB_TITLES[n]}](notebooks/{NB_NAMES[n]}) | {WHAT[n]} | [![Open In Colab]({BADGE})]({colab_url(n)}) |" for n in ON_COURSE_PAGE)
nb_links = "\n".join(f"- {colab_url(n)}" for n in ON_COURSE_PAGE)

readme = f"""# Session 4 · Forecasting With Tree-Based Models

**Advanced Data Analysis Techniques · Master Supply Chain Management · FH OÖ Steyr · Sina Mirshahi**

Course material for session 4: the slides, {'four' if 4 in ON_COURSE_PAGE else 'three'} Colab notebooks and one synthetic dataset. One question carries the session: **what will next week bring, and under which conditions does it change?** You learn one method family in depth, tree-based models (a decision tree, a random forest, LightGBM): when to use them instead of a regression or clustering, how to test a forecast fairly, how to spot faulty inputs and outputs, also in code an AI wrote for you, and how to explain a forecast.

## Slides

[The slides as PDF]({PDF})

## Notebooks

| Notebook | What it is for | Open in Colab |
|---|---|---|
{nb_rows}

If a button does not open, copy the address instead:

{nb_links}

Or open Colab and choose **File, Open notebook, GitHub**, type `{REPO}` and pick the notebook.

**Keep your work:** in Colab choose **File, Save a copy in Drive** before you change anything. The copy goes to your own Google Drive; without it, your changes are lost when you close the tab.

Every cell shows its code. `# EDIT` marks what you may change, `# CHECK` the number you should see. Most cells come with an **Ask an AI** prompt for Gemini, ChatGPT or Claude, and a line on how to check the answer. The models train in Colab in a few seconds; nothing needs to be installed.

## The data

One synthetic dataset, built like a real drinks wholesaler's weekly sales: {D['products']} products, {D['weeks']} weeks from {D['first_week']} to {D['last_week']}, {D['rows']:,} rows, with weather, promotions, holidays, a growing product and a record heat wave. No real company data is contained. [data/README.md](data/README.md) describes every column.

## What the session finds

On every week of 2025, learned from 2023 and 2024 (WAPE: all misses as a percentage of all cases sold):

| Method | WAPE on 2025 |
|---|---|
| Repeat last year (the benchmark) | {RES['Repeat last year']['WAPE']:.1f} % |
| Linear regression, all products | {RES['Linear regression']['WAPE']:.1f} % |
| Decision tree | {RES['Decision tree']['WAPE']:.1f} % |
| Random forest | {RES['Random forest']['WAPE']:.1f} % |
| LightGBM | {RES['LightGBM']['WAPE']:.1f} % |

The trees win where the weather and the promotions bend the sales; for orange juice, whose sales hardly react to the weather, a regression of its own is best in this test ({N2['local']['Orange juice']:.1f} %).

## Your project

If you choose tree-based models for your project, {TEMPLATE} Run it once unchanged on the course data, then change the `# EDIT` lines to your own table, one section at a time. Its sections follow the report criteria of the course:

| Report criterion | Weight | Template section |
|---|---|---|
| Question and SCM relevance | 15 % | 1 · your question and the decision |
| Data preparation and features | 20 % | 2 · load and check · 3 · inputs known in time |
| Method and implementation | 20 % | 4 · three periods: learn, choose, test · 5 · four methods compared on validation |
| Checks and experiments | 25 % | 5 · your experiment on validation · 6 · the test, once, and the five checks |
| Interpretation and presentation | 20 % | 7 · explain the model · 8 · forecast and decision |

Share only data you are allowed to share with Colab and with an AI tool, and say in your report where you used AI and how you checked it.

## How the numbers were made

`analysis/build_evidence.py` executes the code cells of the notebooks on the data and freezes what they print into `evidence/results.json`; the slides and the `# CHECK` lines read that file, so they show exactly what the notebooks compute. To rebuild on your own computer:

```bash
python3 -m pip install -r requirements.txt
python3 data/make_dataset.py              # the data (fixed seed, identical files)
python3 analysis/build_evidence.py        # runs the notebook code, writes evidence/results.json
python3 scripts/build_notebooks.py        # the four notebooks
```
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")

yearly = D["yearly"]
prod_rows = "\n".join(f"| {p} | {D['categories'][p]} | {D['cases'][p]} | {D['pallet'][p]} | {D['product_mean'][p]:,} |" for p in D["product_list"])
data_readme = f"""# Course data: weekly sales of a drinks wholesaler

Synthetic weekly sales of a drinks wholesaler in Upper Austria, built for Session 4, *Forecasting With Tree-Based Models* (Advanced Data Analysis Techniques, FH OÖ Steyr). No real company data is contained. `make_dataset.py` makes the files with a fixed random seed, so running it again gives identical files.

## Files

| File | Rows | One row is |
|---|---|---|
| `drinks_weekly.csv` | {D['rows']:,} | one product in one week ({D['products']} products × {D['weeks']} weeks) |
| `products.csv` | {D['products']} | one product: category, case, cases per pallet, price |
| `weeks.csv` | {D['weeks']} | one week: temperature forecast and actual, public holidays |
| `make_dataset.py` | | the generator (fixed seed) |

Period: **{D['first_week']} to {D['last_week']}**, weeks starting on Monday.

## `drinks_weekly.csv`

| Column | Meaning | Known on the Sunday before the week? |
|---|---|---|
| `week_start` | the Monday of the week | yes |
| `product` | one of eight drinks | yes |
| `units` | cases sold in the week: **the target** | no, it is the answer |
| `pallets` | pallets picked for these cases (units ÷ cases per pallet) | no, only after the week |
| `temp_forecast` | weather forecast for the week: mean of the daily maximum, °C | yes |
| `temp_actual` | the weather that came | no, only after the week |
| `discount_pct` | planned promotion discount in percent, 0 = none (10, 20 or 30) | yes, promotions are planned |
| `holiday_days` | Austrian public holidays from Monday to Friday | yes |

`pallets` and `temp_actual` are in the file on purpose: notebook 3 shows what happens when a model uses them.

## The products

| Product | Category | Case | Cases per pallet | Cases in an average week |
|---|---|---|---|---|
{prod_rows}

## What is in the data on purpose

- **Weather that bends the sales.** Below about 15 °C the summer drinks hardly react; above 20 °C they climb steeply. Each product has its own curve; orange juice does not react at all.
- **Promotions that depend on the weather.** {D['promo_weeks']} promotion weeks ({D['promo_share']:.0f} % of all), planned in advance. A discount lifts the summer drinks more in warm weeks than in cool ones.
- **Advent.** Mulled wine sells from mid November to Christmas, more when it is cold, and almost nothing the rest of the year.
- **Growth.** The energy drink sells more every year: {yearly['Energy drink']['2022']} cases in an average week of 2022, {yearly['Energy drink']['2025']} in 2025.
- **Holidays.** Every public holiday on a weekday costs a delivery day.
- **A record heat wave.** The weeks of 23 June to 7 July 2025 are hotter than any week of 2022 to 2024.
- **A slow drift** in each product's level, so recent weeks carry information.

## Licence

Free to use for teaching. Synthetic data; no real company is described.
"""
(ROOT / "data" / "README.md").write_text(data_readme, encoding="utf-8")
for p in ("README.md", "data/README.md"):
    t = (ROOT / p).read_text(encoding="utf-8")
    assert chr(0x2014) not in t and chr(0x2013) not in t, f"dash in {p}"
print("README.md and data/README.md written")

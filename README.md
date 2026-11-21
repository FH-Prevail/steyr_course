# Session 4 · Forecasting With Tree-Based Models

**Advanced Data Analysis Techniques · Master Supply Chain Management · FH OÖ Steyr · Sina Mirshahi**

Course material for session 4: the slides, three Colab notebooks and one synthetic dataset. One question carries the session: **what will next week bring, and under which conditions does it change?** You learn one method family in depth, tree-based models (a decision tree, a random forest, LightGBM): when to use them instead of a regression or clustering, how to test a forecast fairly, how to spot faulty inputs and outputs, also in code an AI wrote for you, and how to explain a forecast.

## Slides

[The slides as PDF](slides/S4_Forecasting_With_Trees.pdf)

[The slides in the browser](https://fh-prevail.github.io/steyr_course/slides/S4_Forecasting_With_Trees.html), building step by step: right arrow for the next step, down arrow for the next slide, O for an overview.

## Notebooks

| Notebook | What it is for | Open in Colab |
|---|---|---|
| [1 · From A Straight Line To A Tree](notebooks/1_line_to_tree.ipynb) | Hands-on 1: the straight line from session 2 against your first decision tree, for ice tea; how deep a tree should grow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/steyr_course/blob/main/notebooks/1_line_to_tree.ipynb) |
| [2 · Many Trees: Forest And Boosting](notebooks/2_forest_and_boosting.ipynb) | Hands-on 2: a random forest and LightGBM for all eight products, tested on 2025 against a benchmark and the regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/steyr_course/blob/main/notebooks/2_forest_and_boosting.ipynb) |
| [3 · Check And Explain](notebooks/3_check_and_explain.ipynb) | Hands-on 3: check code an AI wrote, see what a tree cannot do, and explain a forecast | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/steyr_course/blob/main/notebooks/3_check_and_explain.ipynb) |

If a button does not open, copy the address instead:

- https://colab.research.google.com/github/FH-Prevail/steyr_course/blob/main/notebooks/1_line_to_tree.ipynb
- https://colab.research.google.com/github/FH-Prevail/steyr_course/blob/main/notebooks/2_forest_and_boosting.ipynb
- https://colab.research.google.com/github/FH-Prevail/steyr_course/blob/main/notebooks/3_check_and_explain.ipynb

Or open Colab and choose **File, Open notebook, GitHub**, type `FH-Prevail/steyr_course` and pick the notebook.

**Keep your work:** in Colab choose **File, Save a copy in Drive** before you change anything. The copy goes to your own Google Drive; without it, your changes are lost when you close the tab.

Every cell shows its code. `# EDIT` marks what you may change, `# CHECK` the number you should see. Most cells come with an **Ask an AI** prompt for Gemini, ChatGPT or Claude, and a line on how to check the answer. The models train in Colab in a few seconds; nothing needs to be installed.

## The data

One synthetic dataset, built like a real drinks wholesaler's weekly sales: 8 products, 209 weeks from 2022-01-03 to 2025-12-29, 1,672 rows, with weather, promotions, holidays, a growing product and a record heat wave. No real company data is contained. [data/README.md](data/README.md) describes every column.

## What the session finds

On every week of 2025, learned from 2023 and 2024 (WAPE: all misses as a percentage of all cases sold):

| Method | WAPE on 2025 |
|---|---|
| Repeat last year (the benchmark) | 29.1 % |
| Linear regression, all products | 21.0 % |
| Decision tree | 21.1 % |
| Random forest | 16.1 % |
| LightGBM | 14.8 % |

The trees win where the weather and the promotions bend the sales; for orange juice, whose sales hardly react to the weather, a regression of its own is best in this test (7.2 %).

## Your project

If you choose tree-based models for your project, you get notebook 4, the project template, from me: write to me and I send it to you. Run it once unchanged on the course data, then change the `# EDIT` lines to your own table, one section at a time. Its sections follow the report criteria of the course:

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

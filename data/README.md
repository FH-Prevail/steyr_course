# Course data: weekly sales of a drinks wholesaler

Synthetic weekly sales of a drinks wholesaler in Upper Austria, built for Session 4, *Forecasting With Tree-Based Models* (Advanced Data Analysis Techniques, FH OÖ Steyr). No real company data is contained. `make_dataset.py` makes the files with a fixed random seed, so running it again gives identical files.

## Files

| File | Rows | One row is |
|---|---|---|
| `drinks_weekly.csv` | 1,672 | one product in one week (8 products × 209 weeks) |
| `products.csv` | 8 | one product: category, case, cases per pallet, price |
| `weeks.csv` | 209 | one week: temperature forecast and actual, public holidays |
| `make_dataset.py` | | the generator (fixed seed) |

Period: **2022-01-03 to 2025-12-29**, weeks starting on Monday.

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
| Mineral water | Water | 6 x 1.5 l | 80 | 2,399 |
| Ice tea | Soft drink | 6 x 1.5 l | 80 | 669 |
| Apple spritzer | Soft drink | 6 x 1.5 l | 80 | 827 |
| Lager beer | Beer | 20 x 0.5 l | 40 | 1,635 |
| Radler | Beer | 20 x 0.5 l | 40 | 453 |
| Energy drink | Soft drink | 24 x 0.25 l | 90 | 300 |
| Orange juice | Juice | 6 x 1 l | 100 | 574 |
| Mulled wine | Seasonal | 6 x 1 l | 100 | 115 |

## What is in the data on purpose

- **Weather that bends the sales.** Below about 15 °C the summer drinks hardly react; above 20 °C they climb steeply. Each product has its own curve; orange juice does not react at all.
- **Promotions that depend on the weather.** 224 promotion weeks (13 % of all), planned in advance. A discount lifts the summer drinks more in warm weeks than in cool ones.
- **Advent.** Mulled wine sells from mid November to Christmas, more when it is cold, and almost nothing the rest of the year.
- **Growth.** The energy drink sells more every year: 150 cases in an average week of 2022, 484 in 2025.
- **Holidays.** Every public holiday on a weekday costs a delivery day.
- **A record heat wave.** The weeks of 23 June to 7 July 2025 are hotter than any week of 2022 to 2024.
- **A slow drift** in each product's level, so recent weeks carry information.

## Licence

Free to use for teaching. Synthetic data; no real company is described.

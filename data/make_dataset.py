# -*- coding: utf-8 -*-
"""
make_dataset.py - synthetic weekly demand of a drinks wholesaler in Upper Austria, built for
Session 4 "Forecasting With Tree-Based Models" (Advanced Data Analysis Techniques, FH OÖ Steyr).

Eight products, 209 weeks (Monday 3 January 2022 to Monday 29 December 2025). Demand depends on
the weather in a way a straight line cannot follow: almost nothing happens below about 15 °C, then
sales climb steeply; the mulled wine does the opposite and sells only in Advent. Promotions lift
more in warm weeks than in cold ones, holidays cost delivery days, the energy drink grows every
year, and June/July 2025 holds a heat wave hotter than any week before it.

No real company data is contained. Run it again at any time: the random seed is fixed, so the
files come out identical.

    python data/make_dataset.py
"""
from datetime import date, timedelta
from pathlib import Path

import argparse

import numpy as np
import pandas as pd

SEED = 2026
HERE = Path(__file__).resolve().parent
WEEKS = pd.date_range("2022-01-03", "2025-12-29", freq="W-MON")       # 209 Mondays

# product, category, case, price per case (EUR), base cases a week, noise, promotions a year,
# promotion strength, extra promotion strength in warm weeks
PALLET = {"Mineral water": 80, "Ice tea": 80, "Apple spritzer": 80, "Lager beer": 40, "Radler": 40,
          "Energy drink": 90, "Orange juice": 100, "Mulled wine": 100}      # cases per pallet
PRODUCTS = [
    ("Mineral water",  "Water",         "6 x 1.5 l",   4.50, 2000, 0.06, 7, 0.20, 0.6),
    ("Ice tea",        "Soft drink",    "6 x 1.5 l",   7.20,  600, 0.08, 9, 0.32, 0.8),
    ("Apple spritzer", "Soft drink",    "6 x 1.5 l",   8.40,  800, 0.07, 7, 0.24, 0.4),
    ("Lager beer",     "Beer",          "20 x 0.5 l", 17.90, 1400, 0.07, 9, 0.24, 0.5),
    ("Radler",         "Beer",          "20 x 0.5 l", 16.90,  450, 0.09, 6, 0.28, 0.8),
    ("Energy drink",   "Soft drink",    "24 x 0.25 l", 21.60,  110, 0.08, 9, 0.38, 0.0),
    ("Orange juice",   "Juice",         "6 x 1 l",     9.60,  550, 0.06, 7, 0.20, 0.0),
    ("Mulled wine",    "Seasonal",      "6 x 1 l",    19.80,  900, 0.12, 2, 0.15, 0.0),
]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def easter(y):
    """Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def public_holidays(years):
    """Austrian public holidays (national list)."""
    out = {}
    for y in years:
        for m, d, name in [(1, 1, "New Year"), (1, 6, "Epiphany"), (5, 1, "Labour Day"), (8, 15, "Assumption"),
                           (10, 26, "National Day"), (11, 1, "All Saints"), (12, 8, "Immaculate Conception"),
                           (12, 25, "Christmas Day"), (12, 26, "St Stephen's Day")]:
            out[date(y, m, d)] = name
        e = easter(y)
        for off, name in [(1, "Easter Monday"), (39, "Ascension Day"), (50, "Whit Monday"), (60, "Corpus Christi")]:
            out[e + timedelta(days=off)] = name
    return out


def temperatures(rng):
    """Weekly mean of the daily maximum temperature in °C, Linz-like climate, with heat waves.
    The heat wave of late June 2025 is hotter than anything in 2022 to 2024 on purpose."""
    doy = WEEKS.dayofyear.to_numpy() + 3                              # the Thursday of each week
    clim = 14.2 + 11.8 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    a = np.zeros(len(WEEKS))
    for t in range(1, len(WEEKS)):
        a[t] = 0.45 * a[t - 1] + rng.normal(0, 3.0)
    temp = clim + a
    waves = {"2022-07-18": [3.5, 4.5], "2023-08-14": [4.0, 4.5, 2.0], "2024-08-05": [3.5, 4.5, 3.0],
             "2025-06-23": [5.0, 7.0, 4.0]}
    for start, bumps in waves.items():
        i0 = WEEKS.get_loc(pd.Timestamp(start))
        for j, b in enumerate(bumps):
            temp[i0 + j] = clim[i0 + j] + b + 0.3 * a[i0 + j]
    train = WEEKS < pd.Timestamp("2025-01-01")
    temp[train] = np.minimum(temp[train], 31.6)                       # 2022 to 2024 stay below the 2025 record
    return np.round(temp, 1)


def promotion_plan(rng, n_per_year, months=None):
    """Planned promotion weeks and their discount: never two weeks in a row."""
    disc = np.zeros(len(WEEKS))
    for y in sorted(set(WEEKS.year)):
        idx = [i for i, w in enumerate(WEEKS) if w.year == y and (months is None or w.month in months)]
        chosen = []
        for i in rng.permutation(idx):
            if len(chosen) == n_per_year:
                break
            if all(abs(i - j) > 1 for j in chosen):
                chosen.append(i)
        for i in chosen:
            disc[i] = rng.choice([10, 20, 30], p=[0.40, 0.45, 0.15])
    return disc


def weather_effect(name, t):
    """How the temperature moves a product's sales: 1.0 is a mild spring week."""
    hot = np.maximum(0, t - 28)
    if name == "Mineral water":
        return 0.80 + 1.10 * sig((t - 21) / 3.0) + 0.05 * hot
    if name == "Ice tea":
        return 0.35 + 2.0 * sig((t - 20) / 3.0) + 0.09 * hot
    if name == "Apple spritzer":
        return 0.70 + 0.90 * sig((t - 20) / 3.5) + 0.03 * hot
    if name == "Lager beer":
        return 0.85 + 0.75 * sig((t - 21) / 3.5) + 0.03 * hot
    if name == "Radler":
        return 0.12 + 1.90 * sig((t - 18) / 3.0) + 0.06 * hot
    if name == "Energy drink":
        return 1.0 + 0.004 * (t - 15)
    return np.ones_like(t)                                            # orange juice and mulled wine: no summer effect


def mulled_wine_season(weeks, t):
    """Mulled wine sells in Advent (Christmas markets), more when it is cold; almost nothing otherwise."""
    out = np.full(len(weeks), np.nan)
    for i, w in enumerate(weeks):
        y = w.year
        xmas = date(y, 12, 24)
        start = date(y, 11, 15)
        d = w.date()
        if start - timedelta(days=6) <= d <= start:                   # the week the markets open
            ramp = 0.45
        elif start < d <= xmas - timedelta(days=6):
            ramp = 1.0
        elif xmas - timedelta(days=6) < d <= xmas:                    # the week of Christmas Eve
            ramp = 0.70
        elif xmas < d <= date(y, 12, 31) or d <= date(y, 1, 5):       # between Christmas and Epiphany
            ramp = 0.12
        else:
            ramp = 0.0
        if ramp:
            cold = np.clip(1 + 0.05 * (6 - t[i]), 0.6, 1.7)
            out[i] = ramp * cold
    return out


def main(seed=SEED, out=HERE):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    hol = public_holidays(range(2021, 2027))
    temp = temperatures(rng)
    temp_fc = np.round(temp + rng.normal(0, 1.4, len(WEEKS)), 1)      # the forecast we had on Sunday
    hdays = np.array([sum(1 for k in range(5) if (w.date() + timedelta(days=k)) in hol) for w in WEEKS])
    holiday_names = [", ".join(hol[w.date() + timedelta(days=k)] for k in range(5) if (w.date() + timedelta(days=k)) in hol)
                     for w in WEEKS]
    years = (WEEKS - pd.Timestamp("2022-01-01")).days.to_numpy() / 365.25
    xmas_week = np.array([(w.date() <= date(w.year, 12, 24) <= w.date() + timedelta(days=6)) for w in WEEKS])
    summer = np.isin(WEEKS.month, [6, 7, 8])
    rows = []
    for (name, cat, case, price, base, noise, n_promo, strength, warm) in PRODUCTS:
        months = list(range(4, 10)) if name == "Radler" else ([11, 12] if name == "Mulled wine" else None)
        if name == "Mulled wine":
            disc = np.zeros(len(WEEKS))
            for y in sorted(set(WEEKS.year)):
                idx = [i for i, w in enumerate(WEEKS) if w.year == y and date(y, 11, 20) <= w.date() <= date(y, 12, 14)]
                for i in rng.choice(idx, size=n_promo, replace=False):
                    disc[i] = rng.choice([10, 20], p=[0.6, 0.4])
        else:
            disc = promotion_plan(rng, n_promo, months)
        level = np.zeros(len(WEEKS))                                  # slow market drift: keeps last weeks informative
        for t in range(1, len(WEEKS)):
            level[t] = 0.85 * level[t - 1] + rng.normal(0, 0.025)
        lift = 1 + strength * (disc / 10) ** 1.2 * (1 + warm * sig((temp - 20) / 3))
        holiday = 1 - 0.10 * hdays
        if name == "Mulled wine":
            season = mulled_wine_season(WEEKS, temp)
            expected = np.where(np.isnan(season), 8.0, base * np.nan_to_num(season)) * np.where(np.isnan(season), 1.0, lift * holiday)
            units = np.where(np.isnan(season), rng.poisson(8.0, len(WEEKS)),
                             expected * np.exp(level + rng.normal(0, noise, len(WEEKS)) - noise ** 2 / 2))
        else:
            seasonal = np.ones(len(WEEKS))
            if name == "Lager beer":
                seasonal *= np.where(summer, 1.08, 1.0) * np.where(xmas_week, 1.30, 1.0)
            if name == "Orange juice":
                seasonal *= 1 + 0.10 * np.cos(2 * np.pi * (WEEKS.dayofyear.to_numpy() - 15) / 365.25)
                seasonal *= np.where(xmas_week, 1.25, 1.0)
            if name in ("Mineral water", "Apple spritzer"):
                seasonal *= np.where(xmas_week, 1.08, 1.0)
            trend = np.exp(0.40 * years) if name == "Energy drink" else 1.0
            expected = base * trend * weather_effect(name, temp) * seasonal * lift * holiday
            units = expected * np.exp(level + rng.normal(0, noise, len(WEEKS)) - noise ** 2 / 2)
        units = np.maximum(0, np.round(units)).astype(int)
        for i, w in enumerate(WEEKS):
            rows.append(dict(week_start=w.date().isoformat(), product=name, units=int(units[i]),
                             pallets=round(int(units[i]) / PALLET[name], 1), temp_forecast=float(temp_fc[i]),
                             temp_actual=float(temp[i]), discount_pct=int(disc[i]), holiday_days=int(hdays[i])))
    df = pd.DataFrame(rows)
    df.to_csv(out / "drinks_weekly.csv", index=False)
    prod = pd.DataFrame([dict(product=p[0], category=p[1], case=p[2], cases_per_pallet=PALLET[p[0]], price_per_case_eur=p[3])
                         for p in PRODUCTS])
    prod.to_csv(out / "products.csv", index=False)
    cal = pd.DataFrame(dict(week_start=[w.date().isoformat() for w in WEEKS], temp_forecast=temp_fc, temp_actual=temp,
                            holiday_days=hdays, holidays=holiday_names))
    cal.to_csv(out / "weeks.csv", index=False)
    print(f"{len(df)} rows: {df['product'].nunique()} products x {len(WEEKS)} weeks, {WEEKS[0].date()} to {WEEKS[-1].date()}")
    print(df.groupby("product")["units"].agg(["mean", "min", "max"]).round(0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    main(a.seed, a.out)

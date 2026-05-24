import os
from pathlib import Path

Path(".matplotlib").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv("RGR3_D_1.csv")
x = data["x"].to_numpy(float)
y = data["y"].to_numpy(float)
n = len(x)
x_star = 67.7863
plots = Path("plots")
plots.mkdir(exist_ok=True)

def score(y_true, y_pred):
    rss = float(np.sum((y_true - y_pred) ** 2))
    tss = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - rss / tss
    rmse = float(np.sqrt(rss / len(y_true)))
    a = 100 * float(np.mean(np.abs((y_true - y_pred) / y_true)))
    return rss, r2, rmse, a

grid = np.linspace(x.min(), x.max(), 300)

coef_lin = np.polyfit(x, y, 1)
fit_lin = lambda z: coef_lin[0] * z + coef_lin[1]

coef_quad = np.polyfit(x, y, 2)
fit_quad = lambda z: coef_quad[0] * z**2 + coef_quad[1] * z + coef_quad[2]

coef_power = np.polyfit(np.log(x), np.log(y), 1)
b_power = coef_power[0]
a_power = np.exp(coef_power[1])
fit_power = lambda z: a_power * z**b_power

summary = []
for name, fit in [("linear", fit_lin), ("quadratic", fit_quad), ("power", fit_power)]:
    y_hat = fit(x)
    rss, r2, rmse, a_percent = score(y, y_hat)
    y_star = float(fit(np.array([x_star]))[0])
    summary.append({"model": name, "rss": rss, "r2": r2, "rmse": rmse, "a_percent": a_percent, "y_star": y_star})

lin_y = fit_lin(x)
residuals = y - lin_y
sxx = float(np.sum((x - x.mean()) ** 2))
sxy = float(np.sum((x - x.mean()) * (y - y.mean())))
b = sxy / sxx
a = y.mean() - b * x.mean()
rss_lin = float(np.sum(residuals**2))
s2 = rss_lin / (n - 2)
s = np.sqrt(s2)
se_b = s / np.sqrt(sxx)
se_a = s * np.sqrt(1 / n + x.mean() ** 2 / sxx)
t_crit = stats.t.ppf(0.975, n - 2)
t_obs = b / se_b
p_value = 2 * (1 - stats.t.cdf(abs(t_obs), n - 2))

plt.figure(figsize=(8, 5), dpi=160)
plt.scatter(x, y, s=28, alpha=0.85)
plt.xlabel("Температура, °F")
plt.ylabel("Число поездок")
plt.title("Диаграмма рассеяния")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(plots / "scatter.png")
plt.close()

plt.figure(figsize=(8, 5), dpi=160)
plt.scatter(x, y, s=24, alpha=0.78, label="Наблюдения")
plt.plot(grid, fit_lin(grid), lw=2, label="Линейная")
plt.plot(grid, fit_quad(grid), lw=2, label="Квадратичная")
plt.plot(grid, fit_power(grid), lw=2, label="Степенная")
plt.xlabel("Температура, °F")
plt.ylabel("Число поездок")
plt.title("Регрессионные модели на фоне данных")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(plots / "models.png")
plt.close()

fig, axes = plt.subplots(3, 1, figsize=(8, 8), dpi=160, sharex=True)
series = [("Линейная", fit_lin(x)), ("Квадратичная", fit_quad(x)), ("Степенная", fit_power(x))]
for ax, (title, y_hat) in zip(axes, series):
    ax.axhline(0, lw=1)
    ax.scatter(x, y - y_hat, s=22, alpha=0.8)
    ax.set_ylabel("e")
    ax.set_title(title)
    ax.grid(alpha=0.25)
axes[-1].set_xlabel("Температура, °F")
plt.tight_layout()
plt.savefig(plots / "residuals.png")
plt.close()

print(pd.DataFrame(summary).round(5))
print(pd.Series({
    "a_linear": a,
    "b_linear": b,
    "rss_linear": rss_lin,
    "s2": s2,
    "s": s,
    "se_a": se_a,
    "se_b": se_b,
    "t_crit": t_crit,
    "t_obs": t_obs,
    "p_value": p_value,
    "a_power": a_power,
    "b_power": b_power,
}).round(5))
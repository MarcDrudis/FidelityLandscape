import pathlib

import matplotlib.pyplot as plt
import numpy as np

directory = pathlib.Path(__file__).parent.resolve()
name = "weird_cuts.npy"
plt.style.use(directory.parent / "plots/plot_style.mplstyle")
# terms = [("Y", -0.95), ("ZZ", 1)]
terms = str([("Y", -0.95), ("XX", 1)])


cuts_data = np.load(
    directory.parent / "moving_minima" / terms / name, allow_pickle=True
).item()

cmap = plt.get_cmap("viridis")
import seaborn as sns

cmap = sns.color_palette("flare", as_cmap=True)
norm = plt.Normalize(cuts_data["times"].min(), cuts_data["times"].max())
line_colors = cmap(norm(cuts_data["times"]))
print("colors", line_colors)

width_document = 510 / 72.27
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(width_document, width_document / 3.2))
# axs = [axs]
count = 0
relevant_times = [0, 1.75, 2, 3.75, 4]
for l, t, c in zip(cuts_data["Landscapes"], cuts_data["times"], line_colors):
    # if t in [0, 1, 2, 4, 6]:
    if t in cuts_data["times"]:
        if t == 4:
            continue
        axs[0].plot(
            cuts_data["cut_samples"] / np.pi,
            l,
            color=c,
            linestyle="-",
            linewidth=1.5 if t in relevant_times else 1,
            alpha=1 if t in relevant_times else 0.5,
            label=rf"$\delta t={t}$" if t in relevant_times else None,
        )
# axs.set_xlabel(r"$\norm{\theta}_{\infty}$")
axs[0].set_xlabel(r"Update Size, $\norm{\bm{\theta}}_{\infty}$")
axs[0].tick_params(axis="x", labelsize=11)
axs[0].set_ylabel(r"Infidelity, $\mathcal{L}(\bm{\theta})$")
axs[0].legend(
    borderpad=0.00001,
)

ax2 = axs[1].twinx()
# for n in [4, 6, 8, 10]:
for n in [10]:
    data_mov = np.load(
        directory.parent / "moving_minima" / f"{terms}/moving_minima_qubits={n}.npy",
        allow_pickle=True,
    ).item()
    ps = [np.linalg.norm(c, np.inf) for c in data_mov["perturbation"]]
    # angles = [
    #     np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    #     for a, b in zip(data_mov["perturbation"][1:], data_mov["perturbation"][:-1])
    # ]
    angles = [
        np.arccos(
            np.dot(data_mov["perturbation"][1], b)
            / (np.linalg.norm(data_mov["perturbation"][1]) * np.linalg.norm(b))
        )
        for b in data_mov["perturbation"][1:]
    ]
    print(angles)
    axs[1].plot(data_mov["times"][:-1], ps[:-1], marker=".", label=f"n={n}")
    # ax2.plot(data_mov["times"][1:], angles, marker="x", label=f"n={n} Angle")

axs[1].set_ylabel(r"Update Size, $\norm{\bm{\theta}}_{\infty}$")
axs[1].tick_params(axis="x", labelsize=11)
axs[1].set_xlabel(r"Normalized Time, $\delta t$")
axs[1].legend(
    borderpad=0.00001,
)
# ax2.legend()

plt.savefig(directory.parent / f"plots/weird_plot.svg")

plt.show()

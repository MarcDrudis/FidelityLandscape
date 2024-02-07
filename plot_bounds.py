import matplotlib.pyplot as plt
import numpy as np

fisher_value = 2 ** (-7)
m = 150
xs2 = np.linspace(0, 1, 300)

normalized_variable = 48 * fisher_value / m**2

convexity_regime = (fisher_value / (2**7 * m**2)) / normalized_variable
xs2_small = np.linspace(0, convexity_regime, 300)

tau = fisher_value**2 / (2**9 * m**2)

first_order = fisher_value * normalized_variable * xs2
second_order = m**2 / 48 * normalized_variable**2 * xs2**2

first_order_small = fisher_value * normalized_variable * xs2_small
second_order_small = m**2 / 48 * normalized_variable**2 * xs2_small**2

y_data = first_order - second_order

# Set a consistent color palette
colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]

plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rc("font", family="Times New Roman")
plt.rcParams.update({"errorbar.capsize": 2})


# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot data1 in the first subplot
axs[0].plot(xs2, first_order - second_order, color=colors[0], label="Lower Bound")
axs[0].plot(xs2, first_order + second_order, color=colors[-2], label="Upper Bound")
axs[0].set_xlabel(r"$\frac{48 \mathcal{F}}{m^2}\|{\theta}\|^2$")
axs[0].set_ylabel("Infidelity")
axs[0].axvline(x=0.5, color="grey", linestyle=":")
# axs[0].axvline(x=convexity_regime, color=colors[-1], linestyle="-")
axs[0].set_xticks([0, 0.5, 1])
axs[0].legend()

alpha = 0.2
axs[2].plot(
    xs2_small,
    first_order_small - second_order_small,
    color=colors[0],
    label="Lower Bound",
)
axs[2].plot(
    xs2_small,
    first_order_small + second_order_small,
    color=colors[-2],
    label="Upper Bound",
)
axs[2].fill_between(
    xs2_small,
    first_order_small + second_order_small,
    first_order_small + second_order_small + tau,
    color=colors[-2],
    alpha=alpha,
)
axs[2].fill_between(
    xs2_small,
    first_order_small - second_order_small,
    (first_order_small - second_order_small - tau).clip(min=0),
    color=colors[0],
    alpha=alpha,
)
axs[2].axhline(y=tau, color="grey", linestyle=":")
axs[2].axvline(x=convexity_regime / 2, color="grey", linestyle=":")
axs[2].set_xlim((0, convexity_regime))
axs[2].set_xlabel(r"$\frac{48 \mathcal{F}}{m^2}\|{\theta}\|^2$")
axs[2].set_ylabel("Infidelity")
axs[2].set_xticks([0, 1 / (24 * 2**7)])
axs[2].legend()
tau = 1e-8
axs[1].plot(
    xs2,
    first_order - second_order,
    color=colors[0],
    label="Lower Bound",
)
axs[1].plot(
    xs2,
    first_order + second_order,
    color=colors[-2],
    label="Upper Bound",
)
axs[1].fill_between(
    xs2,
    first_order + second_order,
    first_order + second_order + tau,
    color=colors[-2],
    alpha=alpha,
)
axs[1].fill_between(
    xs2,
    first_order - second_order,
    (first_order - second_order - tau).clip(min=0),
    color=colors[0],
    alpha=alpha,
)
axs[1].axhline(y=tau, color="grey", linestyle=":")
axs[1].set_xlabel(r"$\frac{48 \mathcal{F}}{m^2}\|{\theta}\|^2$")
axs[1].set_ylabel("Infidelity")
axs[1].set_ylim((0, 1e-7))
axs[1].set_xlim((0, 0.5))
axs[1].legend()

# Create a folder for plots if it doesn't exist
import os

if not os.path.exists("plots"):
    os.makedirs("plots")
plt.show()

# plt.show()
# Save the figure in the "plots" folder
plt.savefig("plots/3landscapes")

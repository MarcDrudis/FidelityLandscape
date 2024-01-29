# import matplotlib.pyplot as plt
# import numpy as np
#
# # Set a consistent color palette
# colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]
#
# "red"
#
# data = np.load(
#     "/home/marc/Documents/Fidelity/FidelityLandscape/data_crossing.npy",
#     # "/home/marc/Documents/Fidelity/FidelityLandscape/degenenerate_data.npy",
#     allow_pickle=True,
# ).item()
#
# plt.rcParams.update({"font.size": 12})
# plt.rc("text", usetex=True)
# plt.rc("font", family="Times New Roman")
# plt.rcParams.update({"errorbar.capsize": 2})
#
# plt.plot(
#     data["Time"], data["Global"], marker=".", color=colors[-2], label="Initial Minima"
# )
# plt.plot(data["Time"], data["Local"], color=colors[0], label="Alternative Minima")
# plt.xlabel("Time")
# plt.ylabel("Infidelity")
# plt.legend()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Set a consistent color palette
colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]

# Load data
dataA = np.load(
    "/home/marc/Documents/Fidelity/FidelityLandscape/data_crossing.npy",
    allow_pickle=True,
).item()
dataB = np.load(
    "/home/marc/Documents/Fidelity/FidelityLandscape/degenenerate_data.npy",
    allow_pickle=True,
).item()

plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rc("font", family="Times New Roman")
plt.rcParams.update({"errorbar.capsize": 2})

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot data1 in the first subplot
axs[0].plot(
    dataA["Time"], dataA["Global"], marker=".", color=colors[-2], label="Initial Minima"
)
axs[0].plot(dataA["Time"], dataA["Local"], color=colors[0], label="Alternative Minima")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Infidelity")
axs[0].legend()
axs[0].set_title("Equivalent parameterizations")

# Plot data2 in the second subplot
# Add your code for data2 here

# Add (a) label to the first subplot
axs[0].text(
    -0.1, 1.05, "(a)", transform=axs[0].transAxes, fontsize=14, fontweight="bold"
)

# Add (b) label to the second subplot
# Add your code to add (b) label to the second subplot
axs[1].plot(
    dataB["Time"], dataB["Global"], marker=".", color=colors[-2], label="Initial Minima"
)
axs[1].plot(dataB["Time"], dataB["Local"], color=colors[0], label="Alternative Minima")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Infidelity")
axs[1].legend()
axs[1].set_title("Equivalent parameterizations")
axs[1].text(
    -0.1, 1.05, "(b)", transform=axs[1].transAxes, fontsize=14, fontweight="bold"
)

# Create a folder for plots if it doesn't exist
import os

if not os.path.exists("plots"):
    os.makedirs("plots")

# Save the figure in the "plots" folder
plt.savefig("plots/comparison_plot.png")

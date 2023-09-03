import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

gcn_RAI = [0.832, 0.765, 0.811, 0.748, 0.812]
sgc_RAI = [0.813, 0.794, 0.813, 0.647, 0.765]

gcn_ADV = [0.832, 0.471, 0.624, 0.733, 0.802]
sgc_ADV = [0.813, 0.353, 0.730, 0.735, 0.765]


fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].bar(
    ["RAW", "ATK", "RobustGCN", "MedianGCN", "Ours"],
    gcn_RAI,
    label="GCN-RAI",
    color=["#f55d44", "#f27405", "#ffc848", "#7aa64e", "#50c4f2"],
)
axs[0, 0].set_xlabel("GCN-RAI")
axs[0, 0].set_xlabel("Accuracy")
axs[0, 0].set_ylim(0.3, 0.9)
axs[0, 0].yaxis.grid(True, linestyle="--")
axs[0, 0].set_xticks([])

axs[0, 1].bar(
    ["RAW", "ATK", "RobustGCN", "MedianGCN", "Ours"],
    sgc_RAI,
    label="GCN-RAI",
    color=["#f55d44", "#f27405", "#ffc848", "#7aa64e", "#50c4f2"],
)
axs[0, 1].set_xlabel("SGC-RAI")
axs[0, 1].set_xlabel("Accuracy")
axs[0, 1].set_ylim(0.3, 0.9)
axs[0, 1].yaxis.grid(True, linestyle="--")
axs[0, 1].set_xticks([])


axs[1, 0].bar(
    ["RAW", "ATK", "RobustGCN", "MedianGCN", "Ours"],
    gcn_ADV,
    label="GCN-ADV",
    color=["#f55d44", "#f27405", "#ffc848", "#7aa64e", "#50c4f2"],
)
axs[1, 0].set_xlabel("GCN-ADV")
axs[1, 0].set_xlabel("Accuracy")
axs[1, 0].set_ylim(0.3, 0.9)
axs[1, 0].yaxis.grid(True, linestyle="--")
axs[1, 0].set_xticks([])


axs[1, 1].bar(
    ["RAW", "ATK", "RobustGCN", "MedianGCN", "Ours"],
    sgc_ADV,
    label="SGC-ADV",
    color=["#f55d44", "#f27405", "#ffc848", "#7aa64e", "#50c4f2"],
)
axs[1, 1].set_xlabel("SGC-ADV")
axs[1, 1].set_xlabel("Accuracy")
axs[1, 1].set_ylim(0.3, 0.9)
axs[1, 1].yaxis.grid(True, linestyle="--")
axs[1, 1].set_xticks([])


labels = ["RAW", "ATK", "RobustGCN", "MedianGCN", "Ours"]
color = ["#f55d44", "#f27405", "#ffc848", "#7aa64e", "#50c4f2"]
patches = [
    mpatches.Patch(color=color[i], label="{:s}".format(labels[i]))
    for i in range(len(color))
]


fig.legend(handles=patches, loc="upper right")

plt.savefig("./test.png")

# import matplotlib.pyplot as plt
# import numpy as np

# df = pd.DataFrame(data = {"Task": ["Rapport (Model)",
# 								   "Analyse af Model",
#                                    "Rapport (Teori)",
#                                    "Teori"],
#                             "Start": [15, 14, 9, 9],
#                             "Duration": [7, 7, 11, 7]})

# fig, ax = plt.subplots(1, figsize=(16,6))
# ax.barh(df.Task, df.Duration, left=df.Start)
# plt.xlabel("Kalenderuge")
# plt.savefig("gannt_chart.png")


# Importing the matplotlib.pyplot
import matplotlib.pyplot as plt


MONTHS = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
TASKS_DURATIONS = {
    "Understanding equations of elasticity": [(0, 3)],
    "Setting up JAX framework": [(0, 3.5)],
    "Testing PINNs/DeepONets": [(1, 2.5)],
    "Thesis writing (theory)": [(1, 2.5)],
    "Thesis writing (experiments)": [(2, 2)],
    "Thesis writing (results)": [(3.5, 1.5)],
    "Thesis writing (discussion/conclusion)": [(4, 1)],
    "Proofreading/finish": [(5, 0.5)]
    }

TASKS = list(TASKS_DURATIONS.keys())

TASKS.reverse()
procs = len(TASKS)

fig, gnt = plt.subplots(1, 1, figsize=(20, 10))
fig.subplots_adjust(left=0.4)

gnt.set_xlim(-0.5, 7.5)
gnt.set_ylim(0, procs+1)

gnt.set_xticks([i+0.5 for i in range(7)], minor=True)
gnt.set_yticks([i+1 for i in range(procs)])

gnt.set_xticklabels([])
gnt.set_xticklabels(MONTHS, fontsize=20, minor=True)
gnt.tick_params(pad=400)    
gnt.set_yticklabels(TASKS, fontsize=20, ha='left')

gnt.grid(zorder=0)
[gnt.broken_barh(intervals, (procs-y_pos-0.3, 0.6), facecolors="tab:blue", zorder=3) for y_pos, (_, intervals) in enumerate(TASKS_DURATIONS.items())]
gnt.axvline(x=5.5, color="black")
fig.text(0.755, 0.9, "Hand-in", fontsize=15)

plt.savefig("gantt_chart.png")
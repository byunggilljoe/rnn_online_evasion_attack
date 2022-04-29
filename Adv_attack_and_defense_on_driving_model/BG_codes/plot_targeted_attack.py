import numpy as np
import matplotlib.pyplot as plt

# load data
adv_plot_list = []

x = np.linspace(0, 20, 20)

for p in ["clairvoyant", "predictive", "greedy"]:
    adv_plot_list.append(np.load(f"tmp/{p}_adv_out_sin.npy"))
    plt.plot(x, adv_plot_list[-1])

lin = np.linspace(0, 4*)
plt.plot(x, np.sin(4*np.pi*x))
plt.savefig("tmp/result_sin.jpg")
plt.close()
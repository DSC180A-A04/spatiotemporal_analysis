import matplotlib.pyplot as plt


def plot(x_cal, y_cal, conformal_intervals, quantiles):

    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_index = [2, 1, 3]

    plt.plot(x_cal, y_cal, label="y_true")

    for i, c in zip(range(len(conformal_intervals)), color_index):
        plt.plot(x_cal,
                 conformal_intervals[:, i],
                 c=cycle_colors[c],
                 label=f"p={quantiles[i]}")

    plt.legend()
    plt.show()
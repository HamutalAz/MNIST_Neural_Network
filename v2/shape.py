import matplotlib.pyplot as plt


def plot(x_ax, y_ax, x_label, y_label):

    # create figure - lines
    plt.rcParams["figure.autolayout"] = True
    plt.xlim(0, max(x_ax)+1)
    plt.ylim(0, max(y_ax)+1)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.plot(x_ax, y_ax)
    # if we want to plot multiple - we can add another plt.plot(x_ax, y_ax)
    plt.show()

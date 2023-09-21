import os
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, data_df):
        self.data_df = data_df
        self.plot_folder = "plots"
        os.makedirs(self.plot_folder, exist_ok=True)

    def draw_scatter_plot(self, x_data, y_data, filename, x_label, y_label, title):
        plt.scatter(x_data, y_data, c="blue", alpha=0.7)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        self._save_plot(filename)

    def draw_box_plot(self, data, filename, x_label, y_label, title):
        data.boxplot()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks([1, 2, 3], ["Mean", "Max", "Min"])
        self._save_plot(filename)

    def _save_plot(self, filename):
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/{filename}.png")
        plt.close()

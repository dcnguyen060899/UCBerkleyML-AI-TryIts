# ts_visualizer.py
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_decomposition(decomposed, title_prefix=''):
        fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(12, 8))
        decomposed.observed.plot(ax=axes[0], title=f'{title_prefix}Observed')
        decomposed.trend.plot(ax=axes[1], title=f'{title_prefix}Trend')
        decomposed.seasonal.plot(ax=axes[2], title=f'{title_prefix}Seasonality')
        decomposed.resid.plot(ax=axes[3], title=f'{title_prefix}Residuals')
        plt.tight_layout()
        plt.show()

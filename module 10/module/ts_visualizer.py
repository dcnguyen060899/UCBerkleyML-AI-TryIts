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

    @staticmethod
    def plot_stl_decomposition(decomposed, title_prefix=''):
        """
        Plots the components of an STL decomposition.
        """
        fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(12, 8))
        axes[0].plot(decomposed.observed, label='Observed')
        axes[1].plot(decomposed.trend, label='Trend')
        axes[2].plot(decomposed.seasonal, label='Seasonal')
        axes[3].plot(decomposed.resid, label='Residual')
        
        axes[0].set_title(f'{title_prefix}Observed')
        axes[1].set_title(f'{title_prefix}Trend')
        axes[2].set_title(f'{title_prefix}Seasonality')
        axes[3].set_title(f'{title_prefix}Residuals')
        
        for ax in axes:
            ax.legend()
        plt.tight_layout()
        plt.show()
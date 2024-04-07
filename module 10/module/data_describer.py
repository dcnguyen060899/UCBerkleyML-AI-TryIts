import pandas as pd

class DataDescriber:
    def __init__(self, df):
        self.df = df

    def describe(self):
        description = {
            'start_date': self.df.index.min().strftime('%Y-%m-%d'),
            'end_date': self.df.index.max().strftime('%Y-%m-%d'),
            'number_of_points': len(self.df),
            'frequency': pd.infer_freq(self.df.index),
            'summary_stats': self.df.describe(),
        }
        return description

    def print_description(self):
        description = self.describe()
        print(f"Data Description:")
        for key, value in description.items():
            if isinstance(value, pd.DataFrame):
                print(f"\n{key.capitalize()}:\n{value}\n")
            else:
                print(f"{key.replace('_', ' ').capitalize()}: {value}")

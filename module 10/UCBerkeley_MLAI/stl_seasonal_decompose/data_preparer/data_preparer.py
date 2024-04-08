# data_preparer.py
import pandas as pd

class DataPreparer:
    def __init__(self, df, date_col, freq):
        self.df = df
        self.date_col = date_col
        self.freq = freq

    def prepare(self):
        if self.df[self.date_col].dtype == float:
            self.df[self.date_col] = self.df[self.date_col].astype(int).astype(str)

        if self.freq == 'M':
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            self.df.set_index(self.date_col, inplace=True)
    
        elif self.freq == 'Y':
            # For yearly data, using PeriodIndex because it represents a span of time (the entire year)
            # rather than a specific point in time. This is ideal for data that aggregates over each year.
            self.df.set_index(pd.PeriodIndex(self.df[self.date_col], freq=self.freq), inplace=True)
            self.df.drop(self.date_col, axis=1, inplace=True)
        
        return self.df
    


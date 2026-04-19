import pandas as pd
import numpy as np
from tensor_talezz_rv import fit, predict

df = pd.DataFrame({
    'a': [1, 2, 3, 4, np.nan, 6], 
    'b': ['x', 'y', 'x', 'z', np.nan, 'y'], 
    'c': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01']), 
    'price': [100, 200, 150, 250, 300, 105]
})

model, report = fit(df, target='price', handle_outliers=True, mode='learn', explain=True)
preds = predict(model, df.head())
print('Predictions:', preds)

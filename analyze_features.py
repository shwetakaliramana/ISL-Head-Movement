import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,'.')
from src.ml.feature_engineering import DatasetBuilder, sample_to_features

df = pd.read_csv('data/raw/synthetic_dataset.csv')
# pick first TILT_LEFT sample
sid = df[df['label']=='TILT_LEFT']['sample_id'].iloc[0]
s = df[df['sample_id']==sid].sort_values('frame')
print(f'Sample id {sid} label=TILT_LEFT')
print(s[['frame','yaw','pitch','roll']].head(10))
feat = sample_to_features(s)
print('\nFeatures shape:', feat.shape)
print('First 10 timesteps (yaw,pitch,roll,dyaw,dpitch,droll,roll_sign_smoothed):')
for i,row in enumerate(feat[:10]):
    print(i, ['{:+.2f}'.format(x) for x in row])

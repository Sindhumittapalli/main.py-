# main.py-
(Main process) python 
from utils import fetch_user_data, extract_features, train_model
import pandas as pd

# 1. Fetch Twitter data (or load existing data)
data = fetch_user_data(['@elonmusk', '@fakeaccount123'])  # sample users

# 2. Feature engineering
features_df = extract_features(data)

# 3. Train or load model
model = train_model(features_df)

# 4. Predict spam/fake
predictions = model.predict(features_df.drop('label', axis=1))
print(predictions)

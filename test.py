# import pandas as pd
# INPUT_CSV = "data/backtest_data/binanceGlobalAccounts_BTC_binance_5m_with_factors_with_labels.csv"
# df = pd.read_csv(
#     INPUT_CSV,
#     parse_dates=["open_time", "close_time"],
# ).sort_values("open_time").reset_index(drop=True)
# # Drop all rows where the label column is NaN
# df_clean = df.dropna(subset=['label_SM_minmax']).reset_index(drop=True)
#
# # Save to new CSV
# output_path = "cleaned_with_labels.csv"
# df_clean.to_csv(output_path, index=False)
#
# print(f"Saved cleaned dataset to: {output_path}")
import tensorflow as tf
print("Physical devices:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices("GPU"))

# quick sanity check of where TF places a tensor
print("Tensor device:", tf.constant([1.0]).device)
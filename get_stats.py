import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from plotter import Plotter

# Replace with the URL of your JSON file
url = "https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json"

df = pd.read_json(url)

print(df.tail())


# 1. Mean Absolute Error (MAE) for the number of corners
mae_corners = mean_absolute_error(df["gt_corners"], df["rb_corners"])

# 2. Root Mean Squared Error (RMSE) for the number of corners
rmse_corners = np.sqrt(mean_squared_error(df["gt_corners"], df["rb_corners"]))

# 3. Mean Absolute Deviation (MAD) for deviation values (mean, max, min) in degrees
mad_mean = np.mean(np.abs(df["mean"] - df["gt_corners"]))
mad_max = np.mean(np.abs(df["max"] - df["gt_corners"]))
mad_min = np.mean(np.abs(df["min"] - df["gt_corners"]))

# 4. Root Mean Squared Deviation (RMSD) for deviation values
rmsd_mean = np.sqrt(np.mean((df["mean"] - df["gt_corners"]) ** 2))
rmsd_max = np.sqrt(np.mean((df["max"] - df["gt_corners"]) ** 2))
rmsd_min = np.sqrt(np.mean((df["min"] - df["gt_corners"]) ** 2))

# 5. Room-wise Statistics (You can print these values for each room)
room_stats = df[["name", "gt_corners", "rb_corners", "mean", "max", "min"]].copy()
room_stats["MAE_corners"] = np.abs(df["gt_corners"] - df["rb_corners"])
room_stats["RMSE_corners"] = (df["gt_corners"] - df["rb_corners"]) ** 2
room_stats["MAD_mean"] = np.abs(df["mean"] - df["gt_corners"])
room_stats["MAD_max"] = np.abs(df["max"] - df["gt_corners"])
room_stats["MAD_min"] = np.abs(df["min"] - df["gt_corners"])
room_stats["RMSD_mean"] = (df["mean"] - df["gt_corners"]) ** 2
room_stats["RMSD_max"] = (df["max"] - df["gt_corners"]) ** 2
room_stats["RMSD_min"] = (df["min"] - df["gt_corners"]) ** 2

# 6. Overall Model Performance
overall_mae_corners = room_stats["MAE_corners"].mean()
overall_rmse_corners = np.sqrt(room_stats["RMSE_corners"].mean())
overall_mad_mean = room_stats["MAD_mean"].mean()
overall_mad_max = room_stats["MAD_max"].mean()
overall_mad_min = room_stats["MAD_min"].mean()
overall_rmsd_mean = np.sqrt(room_stats["RMSD_mean"].mean())
overall_rmsd_max = np.sqrt(room_stats["RMSD_max"].mean())
overall_rmsd_min = np.sqrt(room_stats["RMSD_min"].mean())

# 7. Percentage Accuracy for the number of corners
percentage_accuracy = (df["gt_corners"] == df["rb_corners"]).mean() * 100

# 8. Correlation Coefficient between ground truth and predicted values
correlation_coefficient, _ = pearsonr(df["gt_corners"], df["rb_corners"])

# Print the results
print(f"MAE for the number of corners: {mae_corners:.2f}")
print(f"RMSE for the number of corners: {rmse_corners:.2f}")
print(f"MAD for Mean Deviation: {mad_mean:.2f}")
print(f"MAD for Max Deviation: {mad_max:.2f}")
print(f"MAD for Min Deviation: {mad_min:.2f}")
print(f"RMSD for Mean Deviation: {rmsd_mean:.2f}")
print(f"RMSD for Max Deviation: {rmsd_max:.2f}")
print(f"RMSD for Min Deviation: {rmsd_min:.2f}")
print("\nRoom-wise Statistics:")
print(room_stats)
print("\nOverall Model Performance:")
print(f"Overall MAE for the number of corners: {overall_mae_corners:.2f}")
print(f"Overall RMSE for the number of corners: {overall_rmse_corners:.2f}")
print(f"Overall MAD for Mean Deviation: {overall_mad_mean:.2f}")
print(f"Overall MAD for Max Deviation: {overall_mad_max:.2f}")
print(f"Overall MAD for Min Deviation: {overall_mad_min:.2f}")
print(f"Overall RMSD for Mean Deviation: {overall_rmsd_mean:.2f}")
print(f"Overall RMSD for Max Deviation: {overall_rmsd_max:.2f}")
print(f"Overall RMSD for Min Deviation: {overall_rmsd_min:.2f}")
print(f"Percentage Accuracy for the number of corners: {percentage_accuracy:.2f}%")
print(f"Correlation Coefficient: {correlation_coefficient:.2f}")

plot_drawer = Plotter(df)

plot_drawer.draw_scatter_plot(
    df["gt_corners"],
    df["rb_corners"],
    "Ground_Truth_vs_Predicted_Corner_Counts",
    "Ground Truth Corner Count",
    "Predicted Corner Count",
    "Ground Truth vs. Predicted Corner Counts",
)
plot_drawer.draw_box_plot(
    df[["mean", "max", "min"]],
    "Distribution_of_Deviation_Values",
    "Deviation Types",
    "Deviation in Degrees",
    "Distribution of Deviation Values",
)

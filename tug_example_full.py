import pandas as pd
import tese_kinetikos_func as ktks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing, block selection and trimming
data_path = input('Please insert the path for the TUG sensor data:')
raw_data = pd.read_csv(data_path)
print('Existing TUG test blocks:', raw_data.blockID.unique())
blockID = int(input('Please insert the desired block ID:'))
tug_data_df = raw_data[raw_data['blockID'] == blockID]
tug_data_df = ktks.convert_and_format_timestamp(tug_data_df)
tug_data_trimmed = ktks.trim_timeseries(tug_data_df, ref_col='tms_acc')

# Take user input for the desired alignment column hierarchy
selected_col_hierarchy = input('Please insert the desired alignment column hierarchy (comma-separated if multiple):')
col_hierarchy = selected_col_hierarchy.split(',')
col_hierarchy = [col.strip() for col in col_hierarchy]

aligned_tug_data = ktks.align_timeseries(df=tug_data_trimmed, col_hierarchy=col_hierarchy, strategy='median_delta',
                                         time_df=tug_data_trimmed, ref_col=col_hierarchy[0], print_iter=True)
aligned_tug_data = aligned_tug_data.interpolate(method='linear')
# Sensor data rotating and extraction of the gravity component
timestamps, accelerometer, gyroscope, magnetometer, rot_acc, rot_gyr, rot_mag, rot_m, Q = \
    ktks.rotate_sensors(aligned_tug_data, 'MARG')
global_acc = accelerometer - rot_acc

# Normalization and filtering of sensor data
norm_acc = (global_acc - np.mean(global_acc, axis=0)) / np.std(global_acc, axis=0)

xa_smooth = ktks.butter_lowpass_filter(norm_acc[:, 0], 0.5, 100, 'low')
ya_smooth = ktks.butter_lowpass_filter(norm_acc[:, 1], 0.5, 100, 'low')
za_smooth = ktks.butter_lowpass_filter(norm_acc[:, 2], 0.5, 100, 'low')

# Acceleration signal selection
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.lineplot(x=timestamps, y=xa_smooth, color='blue', label='x')
sns.lineplot(x=timestamps, y=ya_smooth, color='red', label='y')
sns.lineplot(x=timestamps, y=za_smooth, color='green', label='z')
plt.xlabel('Timestamps')
plt.ylabel('Normalized Acceleration (m/s^2)')
plt.title('TUG Acceleration Signals')
plt.legend()
plt.show()
plt.pause(interval=1)
selected_acc = input('Select the acceleration signal (-)(x,y,z):')

if selected_acc == 'x':
    sig_a = xa_smooth
if selected_acc == 'y':
    sig_a = ya_smooth
if selected_acc == 'z':
    sig_a = za_smooth
if selected_acc == '-x':
    sig_a = -xa_smooth
if selected_acc == '-y':
    sig_a = -ya_smooth
if selected_acc == '-z':
    sig_a = -za_smooth

# Gyroscope signal selection (Optional)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.lineplot(x=timestamps, y=np.abs(ktks.cumsum_arr(gyroscope[:, 0])), color='blue', label='x')
sns.lineplot(x=timestamps, y=np.abs(ktks.cumsum_arr(gyroscope[:, 1])), color='red', label='y')
sns.lineplot(x=timestamps, y=np.abs(ktks.cumsum_arr(gyroscope[:, 2])), color='green', label='z')
plt.xlabel('Timestamps')
plt.ylabel('Turn Rate (rad/s)')
plt.title('TUG Cumulative Sum Gyroscope Signals')
plt.legend()
plt.show()
plt.pause(interval=1)
selected_gyr = input('Select the gyroscope signal (x,y,z) or Auto:')

if selected_gyr == 'x':
    gyr_m = 0
elif selected_gyr == 'y':
    gyr_m = 1
elif selected_gyr == 'z':
    gyr_m = 2
else:
    gyr_m = None

# Computing TUG test time
tug = ktks.roi_tug(timestamps=timestamps, gyroscope=gyroscope, accelerometer=None, sig_a=sig_a, percentile_step=7.5,
                   window_size=150, min_number_slopes=20, slope_threshold=0.20, plots=True, graphs=False, times=True,
                   gyr_m=gyr_m)


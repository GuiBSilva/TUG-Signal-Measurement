import pandas as pd
import tese_kinetikos_func as ktks
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importing the aligned signal obtained from METER NOME DO FILE
aligned_signals = pd.read_csv(r'G:\My Drive\Tese\Code e Dados\processed_86906.csv')
aligned_signals = ktks.convert_and_format_timestamp(aligned_signals)

# Rotating sensor data and extracting the gravity component
timestamps, accelerometer, gyroscope, magnetometer, rot_acc, rot_gyr, rot_mag, rot_m, Q = \
    ktks.rotate_sensors(aligned_signals, 'MARG')
global_acc = accelerometer - rot_acc

# Normalization and filtering of sensor data
norm_acc = (global_acc - np.mean(global_acc, axis=0)) / np.std(global_acc, axis=0)

xa_smooth = ktks.butter_lowpass_filter(norm_acc[:, 0], 0.5, 100, 'low')
ya_smooth = ktks.butter_lowpass_filter(norm_acc[:, 1], 0.5, 100, 'low')
za_smooth = ktks.butter_lowpass_filter(norm_acc[:, 2], 0.5, 100, 'low')

# Plotting the acceleration signals in order to select the correct axis. This axis must include the bridge like pattern
# as mentioned in the Materials and Methods section of the thesis. For block 86906 this would be the 'z' axis.

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

# Additionaly, sometimes the gyroscope signal must also be manually selected. In these cases, visual analysis of the
# cumulative sum gyroscope signals must be done. The correct axis is the one which contains a 'spike' roughly in the
# middle of the bridge pattern in the acceleration signal. In the case of block 86906 this would be the 'y' axis.

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

# This stage is always immediately followed by the TUG time measurement stage, however, for the purposes of this
# tutorial a DF containing the timestamps, z axis normalized and smoothed acceleration and the y axis gyroscope signal
# will be saved and used for the next step. The other axes of the gyroscope signal are also saved as the 'roi_tug'
# function requires a 3 dimensional array.

data = {'tms_acc': timestamps, 'z_acc': za_smooth, 'x_gyr': gyroscope[:, 0], 'y_gyr': gyroscope[:, 1],
        'z_gyr': gyroscope[:, 2]}
selected_signals = pd.DataFrame(data)
selected_signals.to_csv('selected_signals.csv', index=False)

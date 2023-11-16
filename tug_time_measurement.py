import pandas as pd
import tese_kinetikos_func as ktks
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Importing the selected signals DF and extracting the necessary columns
selected_signals = pd.read_csv('selected_signals.csv')
selected_signals = ktks.convert_and_format_timestamp(selected_signals)
timestamps = selected_signals.tms_acc
acc = selected_signals.z_acc
gyr = np.stack((selected_signals.x_gyr, selected_signals.y_gyr, selected_signals.z_gyr), axis=-1)

# Computing the TUG test time. Usually the gyr_m (Manual selection of the gyroscope axis) is not necessary, however for
# the purposes of this tutorial, it will be used. The 'gyr_m' value used is '1', corresponding to the y axis.The
# 'percentile_step', 'window_size', 'min_number_slopes' and 'slope threshold' parameters can be adjusted in order to
# modify how the starting and ending points of the TUG test are identified. More information can be found in the
# documentation of the 'roi_tug' function and the Materials and Methods section of the thesis.

tug = ktks.roi_tug(timestamps=timestamps, gyroscope=gyr, accelerometer=None, sig_a=acc, percentile_step=7.5,
                   window_size=75, min_number_slopes=20, slope_threshold=0.20, plots=True, graphs=False, times=True,
                   gyr_m=1)

# The TUG test time, standing/sitting phase indexes and the starting/ending timestamps can be extracted from the output
# of the 'roi_tug' function.

tug_time, first_positive_peak_index, first_negative_peak_index, test_start, test_end = tug

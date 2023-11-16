import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from ahrs.filters import Madgwick
from ahrs import Quaternion
from ahrs.common.orientation import am2q, acc2q
import matplotlib.pyplot as plt
import seaborn as sns
import TUG_functions_kinetikos as ktks

def convert_and_format_timestamp(df):
    """
    Convert timestamp strings in the DataFrame to pandas datetime objects and format them.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing timestamp strings.

    Returns:
        pandas.DataFrame: A DataFrame with the converted and formatted timestamp strings.
    """
    columns = [col for col in df.columns if col.startswith(('tms_'))]
    for col in columns:
        datetime_obj = pd.to_datetime(df[col])
        df[col] = datetime_obj.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        df[col] = pd.to_datetime((df[col]))
    return df


def calculate_deltas(df):
    """
    Calculate time deltas between consecutive values of datetime-like columns in the DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing datetime-like columns.

    Returns:
        pandas.DataFrame: A DataFrame containing the time deltas between consecutive values of the columns.
    """
    columns = [col for col in df.columns if col.startswith(('tms_'))]
    delta_df = pd.DataFrame(columns=columns)
    for col in columns:
        time_delta = (df[col] - df[col].shift()).dt.total_seconds()
        delta_df[col] = time_delta

    return delta_df.loc[1:]


def delta_std_dev(time_df: pd.DataFrame, rnd=6):
    """
    Calculate the average standard deviation of time deltas in the DataFrame.

    Parameters:
        time_df (pandas.DataFrame): The DataFrame containing time-related columns.
        rnd (int): Number of decimal places in the final result.

    Returns:
        float: The average standard deviation of the time deltas rounded to six decimal places.
    """

    time_cols = [col for col in time_df.columns if col.startswith(('tms_'))]
    df_cropped = time_df.copy()[time_cols]
    delta_df = calculate_deltas(df_cropped)

    avg_std_dev = delta_df.std().mean().round(rnd)

    return avg_std_dev


def median_delta(time_df: pd.DataFrame, ref_col: str):
    """
    Calculate the average standard deviation of time deltas in the DataFrame.

    Parameters:
        time_df (pandas.DataFrame): The DataFrame containing time-related columns.
        ref_col (str): Reference column for calcultaing median delta
    Returns:
        float: The average standard deviation of the time deltas rounded to six decimal places.

    """

    df_cropped = pd.DataFrame(time_df.copy()[ref_col])
    delta_df = calculate_deltas(df_cropped)

    median_delta = delta_df.median()

    return median_delta[0]


def fixed_dist(dist: int):
    """
        Returns a fixed distance.

        Parameters:
            dist (int): distance value.

        Returns:
            dist (int): distance value.
        """
    return dist


def find_best_match(val, col, dist):
    """
    Find the index of the best match in the given list 'col' for the specified value 'val'
    within the specified 'distance'.

    Parameters:
        val (int or float or pd.Timestamp): The value for which the best match needs to be found.
        col (list of int or float or pd.Timestamp): The list in which to search for the best match.
        dist (int or float): The maximum allowed difference between 'val' and the elements in 'col'.

    Returns:
        int or None: The index of the best match found in 'col' within the 'distance' from 'val'.
                     If no valid match is found, returns None.
    """

    best_match_idx = None
    best_dif = pd.Timedelta(seconds=dist + 1)
    distance = pd.Timedelta(seconds=dist)
    # Convert col to numpy array for faster operations
    col_array = np.array(col)

    for idx, col_val in enumerate(col_array):
        if pd.notna(val) and pd.notna(col_val):
            dif = abs(pd.Timedelta(val - col_val).total_seconds())
            # Exit early if a perfect match is found
            if dif == 0:
                best_match_idx = idx
                break
            if dif <= distance.total_seconds() and (best_match_idx is None or dif < best_dif.total_seconds()):
                # if the current difference is within the acceptable interval and is smaller than the current best dif
                best_dif = pd.Timedelta(seconds=dif)
                best_match_idx = idx
            elif best_match_idx is not None:
                break

    return best_match_idx


def fill_until_nan(col, aligned_df):
    """
    Fill NaN values in a specified column of a DataFrame up to the first non-NaN value encountered.

    Parameters:
    - col (str): The column name for which NaN values need to be filled.
    - aligned_df (pandas.DataFrame): The aligned DataFrame containing the column to be processed.

    Returns: None
    """
    fill_value = None  # Initialize fill_value for the column
    saved_index = None  # Initialize the saved index for the column
    for index in aligned_df.index:
        value = aligned_df.at[index, col]
        if pd.notna(value):  # Check if the value is not NaN
            fill_value = value  # Set the fill_value when the first non-NaN value is encountered
            saved_index = index  # Save the index
            break  # Exit the inner loop after finding the first non-NaN value
    if fill_value is not None and saved_index is not None:
        # Fill NaN values in the column with fill_value until the saved index
        for index in aligned_df.index:
            if index == saved_index:
                break  # Break the loop when the saved index is reached
            if pd.isna(aligned_df.at[index, col]):
                aligned_df.at[index, col] = fill_value + index * 0.001


def align_timeseries(df, col_hierarchy: list, print_iter=False, strategy='median_delta', **strategy_kwargs):
    """
    Align time series data in the DataFrame based on a specified column hierarchy.

    Parameters:
        df (DataFrame): The input DataFrame containing time series data.
        col_hierarchy (list): A list containing column names representing the hierarchy.
        print_iter (Bool): A boolean indicating if the current iteration is to be printed
        strategy (str, optional): The strategy to calculate distance between values.
                                  Options: 'std_dev' (default), 'fixed'.
        **strategy_kwargs: Additional keyword arguments to be passed to the distance strategy function.

    Returns:
        DataFrame: The aligned DataFrame with data from different columns merged based on the hierarchy.
    """

    aligned_df = pd.DataFrame(columns=df.columns.tolist())
    # Separate the df into sections based on hierarchy
    ref_col = df.columns.get_loc(col_hierarchy[0])
    cols_1 = df.columns[ref_col: ref_col + 4]

    col2 = df.columns.get_loc(col_hierarchy[1])
    cols_2 = df[df.columns[col2: col2 + 4]].copy()
    cols_to_paste2 = cols_2.columns.tolist()

    col3 = df.columns.get_loc(col_hierarchy[2])
    cols_3 = df[df.columns[col3: col3 + 4]].copy()
    cols_to_paste3 = cols_3.columns.tolist()

    common_cols = df.columns[-3:]
    # Fill with first column in hierarchy and common columns
    aligned_df[common_cols] = df[common_cols]
    aligned_df[cols_1] = df[cols_1]

    # Get the distance from the chosen strategy

    strategies = {'std_dev': delta_std_dev,
                  'fixed': fixed_dist,
                  'median_delta': median_delta}

    distance = strategies[strategy](**strategy_kwargs)
    # Go through column one filling with the best matches
    for index, row in aligned_df.iterrows():
        # Extract the value from the first column of the row
        val = row[aligned_df.columns[ref_col]]
        # Find the best match based on the smallest difference between time deltas.
        # If no difference smaller or equal to distance is found Nans are used as values
        best_match2 = find_best_match(val=val, col=cols_2[cols_2.columns[0]], dist=distance)
        best_match3 = find_best_match(val=val, col=cols_3[cols_3.columns[0]], dist=distance)
        # Populate aligned df with the corresponding values from cols 2 and 3. Rows that have been used are
        # dropped, as well as all the rows prior in order to preserve the consistency of the timeseries
        if best_match2 is None:
            row_values_to_paste2 = np.nan
        else:
            row_values_to_paste2 = cols_2.iloc[best_match2].values
            cols_2.drop(index=cols_2.index[:best_match2 + 1], inplace=True)

        if best_match3 is None:
            row_values_to_paste3 = np.nan
        else:
            row_values_to_paste3 = cols_3.iloc[best_match3].values
            cols_3.drop(index=cols_3.index[:best_match3 + 1], inplace=True)

        # Assign values to the corresponding rows in aligned_df

        aligned_df.loc[index, cols_to_paste2] = row_values_to_paste2
        aligned_df.loc[index, cols_to_paste3] = row_values_to_paste3

        # Resetting the indexes in order to properly loop after deletion of rows

        cols_2.reset_index(drop=True, inplace=True)
        cols_3.reset_index(drop=True, inplace=True)

        if print_iter:
            if int(index) % 10 == 0:
                print(index)

    columns_to_drop = [cols_2.columns[0], cols_3.columns[0]]
    aligned_df.drop(columns_to_drop, axis=1, inplace=True)

    # Loop through columns and fill NaN values before the first non-NaN value
    for col in aligned_df.columns:
        fill_until_nan(col, aligned_df)

    # Find the index of the first occurrence of NaT (or NaN)
    first_na_index = aligned_df.index[aligned_df[aligned_df.columns[ref_col]].isna()].min()

    # Trim the DataFrame after the first NaT occurrence (including the NaT row)
    aligned_df = aligned_df.loc[:first_na_index - 1]

    # Typecasting columns to float to allow interpolation
    columns_to_convert = cols_to_paste2 + cols_to_paste3
    columns_to_convert = [col for col in columns_to_convert if not col.startswith('tms')]
    aligned_df[columns_to_convert] = aligned_df[columns_to_convert].astype(float)

    # Get a list of all column names except the reference column
    other_columns = [col for col in aligned_df.columns if col != col_hierarchy[0]]

    # Move the reference column to the leftmost position
    aligned_df = aligned_df[[col_hierarchy[0]] + other_columns]

    return aligned_df


def rotate_sensors(df, method='IMU'):
    """
    Rotate sensor data using either IMU or MARG sensor fusion methods.

    Parameters:
        df (DataFrame): Input DataFrame containing sensor data columns.
        method (str, optional): Sensor fusion method to use. Options: 'IMU' (default), 'MARG'.

    Returns:
        tuple: A tuple containing the following arrays:
            timestamps (Series): Timestamps of the sensor data.
            accelerometer (array): Accelerometer data.
            gyroscope (array): Gyroscope data.
            magnetometer (array): Magnetometer data.
            rot_acc (array): Rotated accelerometer data.
            rot_gyr (array): Rotated gyroscope data.
            rot_mag (array): Rotated magnetometer data.
    """
    # Extract sensor data columns
    accelerometer = np.array(df[['x_acc', 'y_acc', 'z_acc']])
    gyroscope = np.array(df[['x_gyr', 'y_gyr', 'z_gyr']])
    magnetometer = np.array(df[['x_mag', 'y_mag', 'z_mag']])

    # Extract timestamps
    time_col = [col for col in df.columns if col.startswith(('tms_'))]
    timestamps = df[time_col[0]]

    # Initialize Madgwick filter
    madgwick = Madgwick()

    # Initialize arrays
    num_samples = len(timestamps)
    Q = np.zeros((num_samples, 4))
    rot_acc = np.zeros((len(accelerometer), 3))
    rot_gyr = np.zeros((len(gyroscope), 3))
    rot_mag = np.zeros((len(magnetometer), 3))
    rot_m = np.zeros((num_samples, 3, 3))
    # Initialize initial attitude
    if method == 'IMU':
        Q[0] = acc2q(accelerometer[0])
    elif method == 'MARG':
        Q[0] = am2q(accelerometer[0], magnetometer[0])

    # Apply sensor fusion methods
    for t in range(1, num_samples):
        sample_time = (timestamps[t] - timestamps[t - 1]).total_seconds()
        madgwick.Dt = sample_time
        if method == 'IMU':
            Q[t] = madgwick.updateIMU(Q[t - 1], gyr=gyroscope[t], acc=accelerometer[t])
        elif method == 'MARG':
            Q[t] = madgwick.updateMARG(Q[t - 1], gyr=gyroscope[t], acc=accelerometer[t], mag=magnetometer[t])
        rot_matrix = Quaternion(Q[t]).to_DCM()
        rot_m[t] = rot_matrix
        rot_acc[t] = np.dot(rot_matrix, accelerometer[t])
        rot_gyr[t] = np.dot(rot_matrix, gyroscope[t])
        rot_mag[t] = np.dot(rot_matrix, magnetometer[t])

    # Set the first row of rotated data arrays to be the same as the second row
    rot_acc[0] = rot_acc[1]
    rot_gyr[0] = rot_gyr[1]
    rot_mag[0] = rot_mag[1]

    return timestamps, accelerometer, gyroscope, magnetometer, rot_acc, rot_gyr, rot_mag, rot_m, Q


def butter_lowpass_filter(data, cutoff_freq, sampling_freq, filter_type, order=5):
    """
    Applies a Butterworth filter to a given signal using zero-phase filtering.

    Parameters:
        data : The input column containing the signal.
        cutoff_freq(int): The cutoff frequency of the filter.
        sampling_freq(int): The sampling frequency of the original signal.
        filter_type(str): Type of band used. See scipy.signal.butter documentation
                          for accepted types.
        order(int): The order of the filter.
    Returns:
        Column with the filtered signal(s).
    """
    nyquist_freq = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)

    # Apply zero-phase filtering
    filtered_data = filtfilt(b, a, data)

    return filtered_data

def roi_tug(timestamps, gyroscope, gyr_m=None, accelerometer=None, sig_a=None, percentile_step=2.5, window_size=75,
            min_number_slopes=20, der_slope_window=6, slope_threshold=0.1, plots=False, graphs=False, times=False):
    """
    Computes the Time Up and Go (TUG) analysis on a given set of sensor data to determine various events and durations. Check turn_roi, standing_roi and sitting_roi
    for more information on how the RoIs are found.

    Parameters:
        timestamps (list): List of timestamps corresponding to the sensor data.
        gyroscope (list): List of gyroscope data.
        accelerometer (list, optional): List of accelerometer data. Default is None.
        sig_a (list, optional): Signal data. Default is None.
        percentile_step (float, optional): The step size for calculating percentiles. Default is 2.5.
        window_size (int, optional): Size of the sliding window used for analysis. Default is 75.
        min_number_slopes (int, optional): Minimum number of slopes required for analysis. Default is 20.
        der_slope_window (int, optional): Number of slopes in the derivative signal used to calculate the average rate of change of acceleration. Default is 6.
        plots (bool, optional): If True, plot the results. Default is True.
        graphs (bool, optional): If True, generate graphs during analysis. Default is False.

    Returns:
        tug_time (float): Total TUG time in seconds.
        first_positive_peak_index (int): Index of the first positive peak in the signal.
        first_negative_peak_index (int): Index of the first negative peak in the signal.
        test_start (int): Index of the inflection point before the first positive peak.
        test_end (int): Index of the inflection point after the first negative peak.

    """
    if gyr_m is not None:
        turn_index = turn_roi(gyroscope=gyroscope, percentile=percentile_step, graphs=graphs, accelerometer=None,
                              window_size=50, gyr_m=gyr_m)
    else:
        turn_index = turn_roi(gyroscope=gyroscope, percentile=percentile_step, graphs=graphs, accelerometer=None,
                              window_size=50)
    # When (and if) the correct signal can be automatically determined, insert the function here
    # and assign its output to sig_a, then delete sig_a from the parameters
    # Compute the derivative using numpy's gradient function
    time_numerical = np.array([(t - timestamps[0]).total_seconds() / (24 * 3600) for t in timestamps]) * 1E5
    acc_der = np.gradient(sig_a, time_numerical)
    acc_der = ktks.butter_lowpass_filter(acc_der, 1, 100, 'low')
    first_positive_peak_index, test_start = standing_roi(timestamps, turn_index, acc_der=acc_der,
                                                         sig_a=sig_a, percentile=percentile_step,
                                                         window_size=window_size,
                                                         min_number_slopes=min_number_slopes,
                                                         der_slope_window=der_slope_window,
                                                         slope_threshold=slope_threshold,
                                                         graphs=graphs)
    first_negative_peak_index, test_end = sitting_roi(timestamps, turn_index,
                                                      test_start, acc_der=acc_der,
                                                      sig_a=sig_a,
                                                      percentile=percentile_step,
                                                      window_size=window_size,
                                                      min_number_slopes=min_number_slopes,
                                                      der_slope_window=der_slope_window,
                                                      slope_threshold=slope_threshold,
                                                      graphs=graphs)
    if plots:
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 5))

        if test_start is not None:
            plt.scatter(x=timestamps[test_start], y=sig_a[test_start],
                        color='cornflowerblue', edgecolors='black', s=80, marker='*', zorder=3,
                        label='Start')

        if test_end is not None:
            plt.scatter(x=timestamps[test_end], y=sig_a[test_end],
                        color='indianred', edgecolors='black', s=80, marker='*', zorder=3,
                        label='End')

        sns.lineplot(x=timestamps, y=sig_a, zorder=2, label='Acceleration Signal', color='green')
        # sns.lineplot(x=timestamps, y=acc_der, zorder=1, label='Derivative Signal', color='orange')

        ylim = plt.ylim()  # Get the current y-axis limits

        if first_positive_peak_index is not None:
            start_index = max(0, first_positive_peak_index - window_size // 2)
            end_index = min(len(timestamps) - 1, first_positive_peak_index + window_size // 2)

            plt.fill_between(
                timestamps[start_index:end_index + 1],
                ylim[0], ylim[1],  # Use y-axis limits to shade the entire height of the graph
                color='darkblue', label='Standing RoI',
                alpha=0.5)

        if first_negative_peak_index is not None:
            start_index = max(0, first_negative_peak_index - window_size // 2)
            end_index = min(len(timestamps) - 1, first_negative_peak_index + window_size // 2)

            plt.fill_between(
                timestamps[start_index:end_index + 1],
                ylim[0], ylim[1],  # Use y-axis limits to shade the entire height of the graph
                color='darkred', label='Sitting Roi',
                alpha=0.5)

        if turn_index is not None:
            start_index = max(0, turn_index - window_size // 2)
            end_index = min(len(timestamps) - 1, turn_index + window_size // 2)

            plt.fill_between(
                timestamps[start_index:end_index + 1],
                ylim[0], ylim[1],  # Use y-axis limits to shade the entire height of the graph
                color='limegreen', label='turn',
                alpha=0.5)

        plt.xlabel('Timestamps')
        plt.ylabel('Normalized Acceleration (m/s^2)')
        plt.title('Acceleration Signal and Turning, Standing and Sitting RoIs')
        plt.legend()
        plt.tight_layout()
        plt.show()

    tug_time = round((timestamps[test_end - 1] - timestamps[test_start]).total_seconds(),1)
    time_to_turn = round((timestamps[turn_index] - timestamps[test_start]).total_seconds(),1)
    time_after_turn = round((timestamps[test_end - 1] - timestamps[turn_index]).total_seconds(),1)
    tbt_tat = np.round((time_to_turn - time_after_turn) / time_to_turn * 100)
    if times:
        print('Total TUG time:', tug_time, 'seconds')
        print('Time before turn time:', time_to_turn, 'seconds')
        print('Time after turn:', time_after_turn, 'seconds')
        print(f'Time before turn is {tbt_tat}% the size of Time after turn.')

    return tug_time, first_positive_peak_index, first_negative_peak_index, test_start, \
        test_end


def cumsum_arr(arr):
    '''
    Function to compute the cumsum of an array TS.
    It adds values when they are positive or negative.
    It resets when the signal chages.
    '''
    cum_sum = [None] * len(arr)
    cum_sum[0] = arr[0]

    for count, _ in enumerate(arr[1:], start=1):
        if ((arr[count - 1] > 0) & (arr[count] > 0)):
            cum_sum[count] = cum_sum[count - 1] + arr[count]
        elif ((arr[count - 1] < 0) & (arr[count] < 0)):
            cum_sum[count] = cum_sum[count - 1] + arr[count]
        else:
            cum_sum[count] = arr[count]

    return cum_sum


def turn_roi(gyroscope, percentile, timestamps=None, graphs=False, accelerometer=None, window_size=50, gyr_m=None):
    """
    Detects a turn in gyroscope data and returns the index of the detected turn point within the data.

    Parameters:
        gyroscope (numpy.ndarray): 3D gyroscope data array containing angular velocity values for x, y, and z axes.
        percentile (float): The percentile threshold used to detect the turn. A lower percentile may detect smaller turns.
        graphs (bool, optional): If True, plots a graph showing the gyroscope data and the detected turn region.
        accelerometer (numpy.ndarray, optional): 3D accelerometer data array, used for visualization purposes if provided.
        window_size (int, optional): The size of the window around the detected turn point for visualization.

    Returns:
        turn_index (int or None): The index of the detected turn point within the gyroscope data. Returns None if no turn is detected.
    """

    cs_gyr_x = cumsum_arr(gyroscope[:, 0])
    cs_gyr_y = cumsum_arr(gyroscope[:, 1])
    cs_gyr_z = cumsum_arr(gyroscope[:, 2])

    # Calculate the peak values of the cumulative sum arrays
    peak_values = np.array([np.max(np.abs(cs_gyr_x)), np.max(np.abs(cs_gyr_y)), np.max(np.abs(cs_gyr_z))])
    # Find the index of the array with the largest peak
    index_of_largest_peak = np.argmax(peak_values)
    if gyr_m is not None:
        index_of_largest_peak = gyr_m
    # Assign the cumulative sum array with the largest peak to cs_gyr
    if index_of_largest_peak == 0:
        cs_gyr = cs_gyr_x
    elif index_of_largest_peak == 1:
        cs_gyr = cs_gyr_y
    else:
        cs_gyr = cs_gyr_z

    turn_index = None
    # Initialize variables to lock in thresholds
    turn_threshold_locked = False
    tu_prct = 0  # turn percentile

    # Finishes after all indexes are found
    while turn_index is None:
        tu_prct += percentile

        if tu_prct >= 100:  # Ensures loop breaks after percentiles reach limits
            break

        if not turn_threshold_locked:
            gyr_start = int(0.15 * len(cs_gyr))
            gyr_end = int(0.85 * len(cs_gyr))
            turn_threshold = np.percentile(np.abs(cs_gyr[gyr_start:gyr_end]), 100 - tu_prct)  # Setting turn threshold

        # Finding the turn
        turn_index = None
        for i in range(round(0.15 * len(cs_gyr)),
                       round(0.85 * len(cs_gyr))):  # Checks only the middle part of the signal as to avoid
            if np.abs(cs_gyr)[i] >= turn_threshold:  # detecting turns before starting and before sitting
                turn_index = i
                break
        if turn_index is None:
            continue
        # Updates the locked-in turn threshold if needed

        if turn_index is not None:
            turn_threshold_locked = True
    if graphs:
        if accelerometer is not None:
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=timestamps, y=accelerometer,
                         label='Accelerometer', color='green')
            # sns.lineplot(x=timestamps, y=cs_gyr, label='Gyroscope', color='orange')
            ylim = plt.ylim()
        if turn_index is not None:
            start_index = max(0, turn_index - window_size // 2)
            end_index = min(len(timestamps) - 1, turn_index + window_size // 2)

            plt.fill_between(
                timestamps[start_index:end_index + 1],
                ylim[0], ylim[1],  # Use y-axis limits to shade the entire height of the graph
                color='limegreen', label='turn',
                alpha=0.5)

        plt.xlabel('Timestamps')
        plt.ylabel('Normalized Acceleration (m/s^2)')
        plt.title('Acceleration signal and Turning RoI')
        plt.legend()
        plt.tight_layout()
        # plt.show()

    return turn_index


def standing_roi(timestamps, turn_index, acc_der=None, sig_a=None, percentile=2.5, window_size=75, min_number_slopes=20,
                 der_slope_window=6, slope_threshold=0.1, graphs=False):
    """
    Detects the start of a standing event (Sit-to-Stand) and the inflection point before it in a signal. This function limits the signal to the section
    prior to the turn index as to limit the impact of events that are not part of the TUG test.

    Parameters:
        timestamps (list): List of timestamps corresponding to the signal data.
        turn_index (int): Index of the turn event in the signal.
        acc_der (list, optional): Derivative of the signal (acceleration). Default is None.
        sig_a (list, optional): Signal data (acceleration). Default is None.
        percentile (float, optional): The percentile threshold for detecting the start of the standing event. Default is 2.5.
        window_size (int, optional): Size of the sliding window for visualization. Default is 75.
        min_number_slopes (int, optional): Minimum number of consecutive positive slopes required for detection. Default is 20.
        der_slope_window (int, optional): Number of slopes in the derivative signal used to calculate the average rate of change of acceleration.
        slope_threshold (int, optional): Maximum average rate of change for the acceleration signal.
        graphs (bool, optional): If True, generate visualization graphs. Default is False.

    Returns:
        first_positive_peak_index (int): Index of the first positive peak (standing event).
        test_start (int): Index of the inflection point before the start of standing.
    """
    first_positive_peak_index = None
    positive_peak_locked = False
    st_prct = 0  # standing percentile

    while first_positive_peak_index is None:
        st_prct += percentile

        # Finding the index of the first positive peak larger than the positive_peak_threshold
        if not positive_peak_locked:
            positive_peak_threshold = np.percentile(acc_der[0: round(2.5 * turn_index)],
                                                    100 - st_prct)  # setting the threshold

        first_positive_peak_index = None
        positive_slope_count = 0

        for i in range(turn_index):
            if acc_der[i] > positive_peak_threshold:
                for j in range(i, i + min_number_slopes):
                    if j < len(acc_der) and acc_der[j] > acc_der[j - 1]:  # checking if increasing acceleration is
                        positive_slope_count += 1  # kept for a certain number of slopes
                    else:
                        break
                if positive_slope_count >= min_number_slopes:
                    first_positive_peak_index = i
                    break
        # Restart the loop if positive peak threshold is not found
        if first_positive_peak_index is None:
            continue

        # Find the inflection point before the first positive peak
        test_start = 0
        if first_positive_peak_index is not None:
            for i in reversed(range(first_positive_peak_index)):
                if acc_der[i] == 0:
                    test_start = i
                    break
                avg_slope = np.mean(acc_der[round(i - der_slope_window / 2):round(i + der_slope_window / 2)])
                if np.abs(avg_slope) <= slope_threshold:
                    test_start = i
                    break
        if graphs:
            # Plot the original signal and shade the windows around the first positive and negative peaks
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 5))
            # sns.lineplot(x=timestamps, y=ktks.butter_lowpass_filter(accelerometer[:, 0], 0.3, 100, 'low'), label='x')
            sns.lineplot(x=timestamps[0:turn_index], y=acc_der[0:turn_index], label='Derivative of z', color='orange')
            sns.lineplot(x=timestamps[0:turn_index], y=positive_peak_threshold, label=f'{100 - st_prct}th Percentile',
                         color='Black')
            sns.lineplot(x=timestamps[0:turn_index], y=sig_a[0:turn_index], label='z',
                         color='green')
            ylim = plt.ylim()  # Get the current y-axis limits

            if test_start is not None:
                plt.scatter(x=timestamps[test_start], y=sig_a[test_start],
                            color='cornflowerblue', edgecolors='black', s=80, marker='*', zorder=3,
                            label='Start')

            if first_positive_peak_index is not None:
                start_index = max(0, first_positive_peak_index - window_size // 2)
                end_index = min(len(timestamps) - 1, first_positive_peak_index + window_size // 2)

                plt.fill_between(
                    timestamps[start_index:end_index + 1],
                    ylim[0], ylim[1],  # Use y-axis limits to shade the entire height of the graph
                    color='darkblue', label='Standing RoI',
                    alpha=0.5)
            plt.xlabel('Timestamps')
            plt.ylabel('Normalized Acceleration (m/s^2)')
            plt.title('Acceleration signal and Standing RoIs')
            plt.legend()
            plt.tight_layout()
            plt.show()
    return first_positive_peak_index, test_start


def sitting_roi(timestamps, turn_index, test_start, acc_der=None, sig_a=None, percentile=2.5,
                window_size=75, min_number_slopes=20, der_slope_window=5, slope_threshold=0.1, graphs=False):
    """
    Detects the end of a sitting event (Stand-to-Sit) and the inflection point after it in a signal. The function looks at a section of the signal that starts at
    the turn index and has a length equal to 1.3 times the duration of the time to turn (in order to account for subjects where the post-turn phase takes longer).
    The threshold is also calculated for this section only, in order to avoid peaks caused by events outside of the TUG test that might affect thresholding values.

    Parameters:
    timestamps (list): List of timestamps corresponding to the signal data.
    turn_index (int): Index of the turn event in the signal.
    test_start (int): Index of the inflection point before the start of standing.
    acc_der (list, optional): Derivative of the signal (acceleration). Default is None.
    sig_a (list, optional): Signal data (acceleration). Default is None.
    percentile (float, optional): The percentile threshold for detecting the end of the sitting event. Default is 2.5.
    window_size (int, optional): Size of the sliding window for visualization. Default is 75.
    min_number_slopes (int, optional): Minimum number of consecutive negative slopes required for detection. Default is 20.
    der_slope_window (int, optional): Number of slopes in the derivative signal used to calculate the average rate of change of acceleration.
    slope_threshold (int, optional): Maximum average rate of change for the acceleration signal.
    graphs (bool, optional): If True, generate visualization graphs. Default is False.

    Returns:
    first_negative_peak_index (int): Index of the first negative peak (sitting event).
    test_end (int): Index of the inflection point after the end of sitting.
    """
    sig_length = turn_index - test_start
    max_sig = turn_index + round(1.3 * sig_length) if turn_index + round(1.3 * sig_length) <= len(acc_der) else len(
        acc_der)
    min_sig = turn_index + round(0.3 * sig_length) if turn_index + round(0.3 * sig_length) <= len(acc_der) else len(
        acc_der)

    first_negative_peak_index = None
    si_prct = 0  # standing percentile

    while first_negative_peak_index is None:
        si_prct += percentile

        negative_peak_threshold = np.percentile(acc_der[min_sig: max_sig], si_prct)

        first_negative_peak_index = None
        negative_slope_count = 0

        for i in reversed(range(turn_index + round(0.3 * sig_length), max_sig)):
            if acc_der[i] < negative_peak_threshold:
                for j in range(i, i + min_number_slopes):
                    if j < len(acc_der) and acc_der[j] < acc_der[j - 1]:  # checking if decreasing acceleration
                        negative_slope_count += 1  # is kept for a certain number of samples
                    else:
                        break
                if negative_slope_count >= min_number_slopes:
                    first_negative_peak_index = i
                    break

        # Find the inflection point after the first negative peak
        test_end = len(sig_a)
        if first_negative_peak_index is not None:
            for i in range(first_negative_peak_index + 1, len(acc_der) - 1):
                if sig_a[i - 1] >= sig_a[i] < sig_a[i + 1]:
                    test_end = i
                    break
                avg_slope = np.mean(acc_der[round(i - der_slope_window / 2):round(i + der_slope_window / 2)])
                if np.abs(avg_slope) <= slope_threshold:
                    test_end = i
                    break
        if graphs:
            # Plot the original signal and shade the windows around the first positive and negative peaks
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 5))
            # sns.lineplot(x=timestamps, y=ktks.butter_lowpass_filter(accelerometer[:, 0], 0.3, 100, 'low'), label='x')
            sns.lineplot(x=timestamps[min_sig: max_sig], y=acc_der[min_sig: max_sig], label='Derivative of z',
                         color='orange')
            sns.lineplot(x=timestamps[min_sig: max_sig], y=negative_peak_threshold, label=f'{si_prct}th Percentile',
                         color='Black')
            sns.lineplot(x=timestamps[min_sig: max_sig], y=sig_a[min_sig: max_sig], label='z',
                         color='green')

            ylim = plt.ylim()  # Get the current y-axis limits

            if test_end is not None:
                plt.scatter(x=timestamps[test_end], y=sig_a[test_end],
                            color='indianred', edgecolors='black', s=80, marker='*', zorder=3,
                            label='End')

            if first_negative_peak_index is not None:
                start_index = max(0, first_negative_peak_index - window_size // 2)
                end_index = min(len(timestamps) - 1, first_negative_peak_index + window_size // 2)

                plt.fill_between(
                    timestamps[start_index:end_index + 1],
                    ylim[0], ylim[1],  # Use y-axis limits to shade the entire height of the graph
                    color='darkred', label='Sitting Roi',
                    alpha=0.5)
            plt.xlabel('Timestamps')
            plt.ylabel('Normalized Acceleration (m/s^2)')
            plt.title('Acceleration signal and Sitting RoIs')
            plt.legend()
            plt.tight_layout()
            plt.show()
    return first_negative_peak_index, test_end


def bland_altman_plot(m1, m2, title):
    difs = []
    avg = []
    m1 = m1
    m2 = m2
    for i in range(len(m1)):
        dif = m1[i] - m2[i]
        difs.append(dif)
        average = (m1[i] + m2[i])/2
        avg.append(average)
    mean_dif = np.mean(difs)
    m_dif = [mean_dif]*len(avg)

    standard_error = np.std(difs, axis=0)

    lower_limit_of_agreement = mean_dif - 1.96 * standard_error
    upper_limit_of_agreement = mean_dif + 1.96 * standard_error
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))

    # Scatter plot
    sns.scatterplot(x=avg, y=difs, color= 'cornflowerblue')

    # Line plot with the same xlim as the scatter plot
    plt.axhline(y=mean_dif, color='black', label='Mean Difference')
    plt.axhline(y=lower_limit_of_agreement, color= 'indianred', linestyle= '--', label='Lower LoA')
    plt.axhline(y=upper_limit_of_agreement, color= 'indianred', linestyle= '--', label='Upper LoA')
    # Set the x-axis limits (xlim) to match the scatter plot
    plt.legend()
    plt.xlim(round(min(avg), 0)-0.2, round(max(avg), 0)+0.2)
    plt.ylim(lower_limit_of_agreement-0.2, upper_limit_of_agreement+0.2)
    plt.ylabel('Differences (s)')
    plt.xlabel('Means (s)')
    plt.title('Bland-Altman Plot ' + title)
    plt.show()
    print('Upper LoA', round(upper_limit_of_agreement, 1))
    print('Lower LoA', round(lower_limit_of_agreement, 1))
    print('Mean Difference', round(mean_dif, 1))

    return mean_dif, upper_limit_of_agreement, lower_limit_of_agreement

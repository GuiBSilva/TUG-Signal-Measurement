import pandas as pd
import TUG_functions_kinetikos as ktks
import numpy as np
# Importing raw data
raw_data_df = pd.read_csv('tug_example_data.csv')
# Initial formatting and test selecting. Block Id must be selected beforehand
test_block = raw_data_df[raw_data_df['blockID'] == 86906]
test_block = ktks.convert_and_format_timestamp(test_block)
test_block = ktks.trim_timeseries(test_block, 'tms_acc')
# Alignment of sensor data and filling in of missing data
col_hierarchy = ['tms_acc', 'tms_gyr', 'tms_mag']
aligned_test_block = ktks.align_timeseries(df=test_block, col_hierarchy=col_hierarchy, strategy='median_delta',
                                   time_df=test_block, ref_col=col_hierarchy[0])
aligned_test_block = aligned_test_block.interpolate(method='linear')
# Saving the resulting DataFrame for future use
aligned_test_block.to_csv('tug_86906_aligned.csv', index=False)
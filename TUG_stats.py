import pandas as pd
import tese_kinetikos_func as ktks
from sklearn.model_selection import train_test_split

result_df = pd.read_csv(r'G:\My Drive\Tese\Code e Dados\Time Tests FCUL\gui_vitor_fcul_tug_compared_tug_t020.csv')
result_df['MeanCrono'] = round((result_df.Cronometer1 + result_df.Cronometer2)/2, 2)
result_df['MeanLabel'] = round((result_df.MTT1 + result_df.MTT2)/2, 2)
result_df['AdjustedATT'] = result_df.ATT*0.4683 + 4.8399
train_df, test_df = train_test_split(result_df, test_size=0.25, random_state=0)
train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)

# B-A plot for mean Chronometer and mean Signal raters
print('Bland-Altman plot values for Chronometer and Signal measurements')
ktks.bland_altman_plot(result_df.MeanCrono, result_df.MeanLabel, 'for Chronometer and Signal measurements')

# B-A plot for mean Signal and Automated raters
print('Bland-Altman plot values for Signal and Automated measurements')
ktks.bland_altman_plot(result_df.MeanLabel, result_df.ATT, 'for Signal and Automated measurements')

# B-A plot for mean Chronometer and Automated raters
print('Bland-Altman plot values for Chronometer and Automated measurements')
ktks.bland_altman_plot(result_df.MeanCrono, result_df.ATT, 'for Chronometer and Automated measurements')

# B-A plot for mean Chronometer and Adjusted Automated raters
print('Bland-Altman plot values for Chronometer and Automated measurements')
ktks.bland_altman_plot(result_df.MeanCrono, result_df.AdjustedATT, 'for Chronometer and Adjusted Automated measurements')

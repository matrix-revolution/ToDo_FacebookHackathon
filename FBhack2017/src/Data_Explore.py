import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')

# df_data = pd.read_csv('../bin/Input/Incidentdata.txt', sep='\t')
# l_col = [u'RMS CDW ID', u'General Offense Number', u'Offense Code',
#          u'Offense Code Extension', u'Offense Type', u'Summary Offense Code',
#          u'Summarized Offense Description', u'Date Reported',
#          u'Occurred Date or Date Range Start', u'Occurred Date Range End',
#          u'Hundred Block Location', u'District/Sector', u'Zone/Beat',
#          u'Census Tract 2000', u'Longitude', u'Latitude', u'Location', u'Month',
#          u'Year']
# l_col_sel = [u'Summary Offense Code',
#              u'Summarized Offense Description',
#              u'Occurred Date or Date Range Start',
#              u'District/Sector', u'Zone/Beat',
#              u'Longitude', u'Latitude',
#              u'Month', u'Year']
#
# df_data = df_data.loc[:,l_col_sel]
# df_data.columns=['OffenseCode', 'OffenseDescrip',
#                  'OccurDate', 'DistrictID', 'ZoneID',
#                  'Longitude', 'Latitude',
#                  'Month', 'Year']
# df_data.OccurDate = pd.to_datetime(df_data.OccurDate)
#
# df_data.to_pickle('../bin/Input/Incidentdata.pkl')

###
df_data = pd.read_pickle('../bin/Input/Incidentdata.pkl')

# Clean data
df_data = df_data.loc[df_data.OccurDate>='2008-01-01']
df_data = df_data.loc[df_data.OccurDate<='2017-08-01']

df_data.sort_values(by=['OccurDate'], inplace=True)
df_data = df_data.loc[df_data.OccurDate.dt.year==df_data.Year]
df_data['Month'] = df_data.OccurDate.dt.month

grp_data = df_data.groupby(['OffenseCode','Year', 'Month', 'DistrictID', 'ZoneID'])
df_grp = grp_data['OffenseDescrip'].count()

df_grp_sea = df_data.groupby(['Year', 'Month'])['OffenseCode'].count().reset_index()
df_grp_sea['Year'] = [str(ii) for ii in df_grp_sea['Year']]
df_grp_sea['Month'] = [str(ii) for ii in df_grp_sea['Month']]

df_grp_sea['Date'] = [str(df_grp_sea['Year'][ii]+'-'+df_grp_sea.Month[ii]+'-01') for ii in df_grp_sea.index]
df_grp_sea['Date'] = pd.to_datetime(df_grp_sea['Date'], format='%Y-%m-%d')
df_grp_sea = df_grp_sea.loc[:,['Date', 'OffenseCode']]
df_grp_sea.columns = ['Date', 'NumberOccur']
df_grp_sea.sort_values(by='Date', inplace=True)

""" Highest Level Forecast """

""" Prepare Time Series """
index = pd.date_range(start=df_grp_sea.Date[0], end=df_grp_sea.Date[len(df_grp_sea)-1], freq='MS')
if len(df_grp_sea)<>len(index):
    print('missing data in input')

import statsmodels.api as sm #tsa.statespace as ss

df_single_errors = pd.DataFrame()
df=df_grp_sea.set_index('Date')
try:
    sa_mdl = sm.tsa.statespace.SARIMAX(df, trend='ct', order=(1,1,1), seasonal_order=(1,1,0,12)).fit(disp=True)
    df_tmp = sa_mdl.filter_results.standardized_forecasts_error[0]
    df_single_errors['SARIMA'] = df_tmp
    #
    # fig = sa_mdl.plot_diagnostics()
    # fig.set_size_inches(10,10)
    # fig.savefig(s_path + 'Output/US19PLs/SARIMA/Diagnostics/'+segID+saveName+'_sarima_diagnostics.png')
    #
    pred_dynamic = sa_mdl.get_prediction(start=len(df), end=len(df)+24, dynamic=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    ax = df.plot(label='observed', figsize=(10,15))
    pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

    ax.fill_betweenx(ax.get_ylim(), df.index[-1], pred_dynamic_ci.index[-1], alpha=.1, zorder=-1)
    ax.set_xlabel('Date')
    ax.set_ylabel('NumberOccur')

    ax.get_figure().savefig('../bin/Figures/sea_all_sarima_forecast.png')
except Exception, e:
    print e


""" Plots """
df_data['Year'] = [str(ii) for ii in df_data['Year']]
df_data['Month'] = [str(ii) for ii in df_data['Month']]
df_data['Date'] = [str(df_data['Year'][ii]+'-'+df_data.Month[ii]+'-01') for ii in df_data.index]
df_data['Date'] = pd.to_datetime(df_data['Date'], format='%Y-%m-%d')

grp_data = df_data.groupby(['Date', 'DistrictID'])['OffenseDescrip'].count()
grp_data = df_data.groupby(['OffenseCode','Date'])['OffenseDescrip'].count()

ax = grp_data.unstack(['DistrictID']).plot().get_figure()
ax.savefig('test.png')


""" Restart """
drop_date = pd.to_datetime('2016-09-01')
df_data = pd.read_pickle('../bin/Input/Incidentdata.pkl')

# Clean data
df_data = df_data.loc[df_data.OccurDate>=drop_date]
df_data = df_data.loc[df_data.OccurDate<'2017-09-01']

df_data.sort_values(by=['OccurDate'], inplace=True)
df_data = df_data.loc[df_data.OccurDate.dt.year==df_data.Year]
df_data['Month'] = df_data.OccurDate.dt.month
df_data['Year'] = [str(ii) for ii in df_data['Year']]
df_data['Month'] = [str(ii) for ii in df_data['Month']]
df_data['Month'] = [str(ii) for ii in df_data['Month']]
df_data['Date'] = [str(df_data['Year'][ii]+'-'+df_data.Month[ii]+'-01') for ii in df_data.index]
df_data['Date'] = pd.to_datetime(df_data['Date'], format='%Y-%m-%d')

""" Daily model """
df_grp_daily = df_data.groupby(['Date', 'DistrictID'])['OffenseDescrip'].count()
ax = df_grp_daily.unstack(['DistrictID']).plot().get_figure()
ax.savefig('test3.png')

df_data['OccurDate'] = pd.DatetimeIndex(df_data['OccurDate']).normalize()
df_grp_daily_input = df_data.reset_index().groupby('OccurDate')['OffenseDescrip'].count()

df_grp = df_grp_daily_input.reset_index()
df_grp.columns = ['Date', 'Amount']

df_single_errors = pd.DataFrame()
df=df_grp.set_index('Date')
try:
    sa_mdl = sm.tsa.statespace.SARIMAX(df, trend='ct', order=(1,1,1), seasonal_order=(1,1,0,7)).fit(disp=True)
    df_tmp = sa_mdl.filter_results.standardized_forecasts_error[0]
    df_single_errors['SARIMA'] = df_tmp
    #
    # fig = sa_mdl.plot_diagnostics()
    # fig.set_size_inches(10,10)
    # fig.savefig(s_path + 'Output/US19PLs/SARIMA/Diagnostics/'+segID+saveName+'_sarima_diagnostics.png')
    #
    pred_dynamic = sa_mdl.get_prediction(start=len(df), end=len(df)+14, dynamic=False)
    pred_dynamic_ci = pred_dynamic.conf_int()
    ax = df.plot(label='observed', figsize=(10,15))
    pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

    ax.fill_betweenx(ax.get_ylim(), df.index[-1], pred_dynamic_ci.index[-1], alpha=.1, zorder=-1)
    ax.set_xlabel('Date')
    ax.set_ylabel('NumberOccur')

    ax.get_figure().savefig('test6.png')
except Exception, e:
    print e


""" Final try """

from sklearn.metrics import mean_squared_error

df_test = pd.read_pickle('../bin/Input/Incidentdata.pkl')

# Clean data
df_test = df_test.loc[df_test.OccurDate>='2017-09-01']
df_test = df_test.loc[df_test.OccurDate<'2017-10-01']
df_test.sort_values(by=['OccurDate'], inplace=True)
df_test = df_test.loc[df_test.OccurDate.dt.year==df_test.Year]
df_test['Month'] = df_test.OccurDate.dt.month
df_test['Year'] = [str(ii) for ii in df_test['Year']]
df_test['Month'] = [str(ii) for ii in df_test['Month']]
df_test['Month'] = [str(ii) for ii in df_test['Month']]
df_test['Date'] = [str(df_test['Year'][ii]+'-'+df_test.Month[ii]+'-01') for ii in df_test.index]
df_test['Date'] = pd.to_datetime(df_test['Date'], format='%Y-%m-%d')
df_test['OccurDate'] = pd.DatetimeIndex(df_test['OccurDate']).normalize()

df_grp_test = df_test.reset_index().groupby('OccurDate')['OffenseDescrip'].count()
fcst = sa_mdl.get_forecast(30).predicted_mean
act = df_grp_test.head(30)
mean_squared_error(act, fcst)

df_grp.to_csv('df_grp.txt', sep='\t')
# test = pd.merge(act,fcst, how='left', left_index=True, right_index=True)
fcst.to_csv('fcst.txt', sep='\t')
act.to_csv('act.txt', sep='\t')
# grp_data = df_data.groupby(['OffenseCode','Year', 'Month', 'DistrictID'])
# df_grp = grp_data.count()





print('test')
from joblib.logger import PrintTime
import pmdarima as pm
import numpy as np
import pandas as pd
import pymssql
from matplotlib import pyplot as plt

#載入資料庫
cnxn =pymssql.connect(host='iotlab.servehttp.com' ,database = 'SmartPowerMonitor',user="*****",password="******$")
#讀取資料表特定欄位資訊並篩選
sql='select * from ElectM1Record WHERE [kW]>=150'
#讀取資料
df = pd.read_sql(sql,cnxn)

model = pm.auto_arima(df.kW, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'   
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)


n_periods = 36 #預測的樣本數量
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df.kW), len(df.kW)+n_periods)

fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot  如果不要圖表 就把Plt都隱藏
plt.plot(df.kW)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()

print(fc_series)
#fc_series為預測結果

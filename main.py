import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split

# مجموعه داده نمونه شامل تاریخ، سال، روز هفته، روز ماه، تعطیلات و فروش را وارد میکنیم
# (SaleAmount ریال فروش)
df = pd.read_csv('DailySalesReport.csv', delimiter=',', index_col='date')

dates = df.index
# جداسازی هدف و ویژهگی ها
target = df.iloc[:, 6]
features = df.iloc[:, 0:6]
print(features)
print(target)

# جداسازی داده آموزشی و داده تستی در این نمونه 10 دصد از داده برای تست انتخاب شده است
features_train, features_test, target_train, target_test = (
    train_test_split(features, target, test_size=0.1, shuffle=True))

# تعداد ستون های ویژهگی ها
features_num_columns = len(features.columns)

# تعریف مدل
model = Sequential()

model.add(Dense(300, activation='relu', input_dim=features_num_columns))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adamW', loss='mean_squared_error')
print("Model Created")

# متناسب کردن مدل با داده های آموزشی
model.fit(features_train, target_train, epochs=100, batch_size=100)
print("Training completed")

# ذخیره مدل آموزشی
model.save("Sales_model.keras")
print("Sales_model.keras saved model to disk in ", os.getcwd())

#  فروش روزانه شناخته شده را برای بررسی نتایج پیش بینی میکنیم
predictions = model.predict(features)
predictions_list = map(lambda x: x[0], predictions)
predictions_series = pd.Series(predictions_list, index=dates)
dates_series = pd.Series(dates)

# وارد کردن داده ها برای پیش بینی
df_newDates = pd.read_csv('forcast_dates.csv', delimiter=',', index_col='date')
print("forcast_dates imported")

# پیش بینی فروش آتی با استفاده از مدل آموزش دیده و تاریخ های آتی وارد شدخ
Predicted_sales = model.predict(df_newDates)

# ذخیره فایل پیشبینی
new_dates_series = pd.Series(df_newDates.index)
new_predictions_list = map(lambda x: x[0], Predicted_sales)
new_predictions_series = pd.Series(new_predictions_list, index=new_dates_series)
new_predictions_series.to_csv("predicted.csv")

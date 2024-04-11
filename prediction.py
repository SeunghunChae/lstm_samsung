import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr
import yfinance as yf

#lstm모형 (나선형 모형)을 사용하여 주가 예측 (keras)
yf.pdr_override()

now=datetime.now()
before = now - relativedelta(years=10)

now_day=now.strftime("%Y-%m-%d")
before_day=before.strftime("%Y-%m-%d")
print(f"end : {now_day}")
print(f"start : {before_day}")

samsung_stock = pdr.get_data_yahoo('005930.KS', start=before_day,end=now_day)
print(samsung_stock)

#예측
close_prices = samsung_stock['Close'].values
print(close_prices)

#30일마다 종가를 1개씩 추정한다.
windown_size = 30

result_list = []
for i in range(len(close_prices)-(windown_size+1)):
    result_list.append(close_prices[i: i + (windown_size +1)])

normal_data=[]
for window in result_list:
    window_list = [((float(p) / float(windows[0]))) for p in window]
    normal_data.append(window_list)

result_list = np.array(normal_data)
print(result_list.shape[0], result_list.shape[1])

#학습용 데이터와 검증용 데이터를 분리한다.
row = int(round(result_list.shape[0] * 0.9))
train = result_list[:row, :]

x_train=train[:, :-1]
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
y_train=train[:, -1]

x_test=result_list[row:,:-1]
x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))
y_test=result_list[row:-1]

print(x_train.shape)
print(x_test.shape)

#예측구성 모델 생성 (only cpu)
model = Sequential()
model.add(LSTM(windown_size, return_sequences=True, input_shape=(windown_size,1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1,activation='Linear'))
model.compile(loss='mse', optimizer='rmsprop') #제곱 평균 오차로 추정. rmsprop 사용 (relu 아님)
model.summary()

#결과파일 생성
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=10,
          epochs=10) #epoch는 그때그때 조정함 (학습횟수)

model.save(r'c:\samsung.h5')

#실제값과 예측값을 그려줌
pred = model.predict(x_test)

pred_price = []
for i in pred:
    pred_prices.append((i+1)*window[0])

real_price=[]
for i in y_test:
    real_price.append((i+1)*window[0])

#차트 그리기
fig=plt.figure(facecolor='white', figsize=(70,15))
ax=fig.add_subplot(234)
ax.plot(real_price, label='real')
ax.plot(pred_price, label='pred')
ax.legend() #예측가기 떄문에 동일 범주로 하나만 쓴다
plt.show()

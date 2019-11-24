import numpy as np
import pandas as pd
from torch import nn,tensor

"""
pandas에서 DataFrame에 적용되는 함수들
sum() 함수 이외에도 pandas에서 DataFrame에 적용되는 함수는 다음의 것들이 있다.
count 전체 성분의 (NaN이 아닌) 값의 갯수를 계산
min, max 전체 성분의 최솟, 최댓값을 계산
argmin, argmax 전체 성분의 최솟값, 최댓값이 위치한 (정수)인덱스를 반환
idxmin, idxmax 전체 인덱스 중 최솟값, 최댓값을 반환
quantile 전체 성분의 특정 사분위수에 해당하는 값을 반환 (0~1 사이)
sum 전체 성분의 합을 계산
mean 전체 성분의 평균을 계산
median 전체 성분의 중간값을 반환
mad 전체 성분의 평균값으로부터의 절대 편차(absolute deviation)의 평균을 계산
std, var 전체 성분의 표준편차, 분산을 계산
cumsum 맨 첫 번째 성분부터 각 성분까지의 누적합을 계산 (0에서부터 계속 더해짐)
cumprod 맨 첫번째 성분부터 각 성분까지의 누적곱을 계산 (1에서부터 계속 곱해짐)
"""
#train_data = np.loadtxt("./data/train.csv", delimiter = ",")
#train_data = pd.read_csv("./data/train.csv")
#iloc과 loc 사용해서 추출
#iloc --> 인덱스로 접근 가능 loc --> 키워드 접근도 가능

def processing(path = "./data/train.csv", training = True, date_change = True, input_var = ['in_out','latitude','longitude','6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride',
       '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff', '8~9_takeoff',
       '9~10_takeoff', '10~11_takeoff', '11~12_takeoff','weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6', 'dis_jejusi', 'dis_seoquipo']):

    data = pd.read_csv(path)

    #Data 변수 변환
    if date_change:
        data["date"] = pd.to_datetime(data["date"])
        data["weekday"] = data["date"].dt.weekday
        
        data = pd.get_dummies(data, columns = ["weekday"])

    #시외 = 1, 시내 = 0
    data.loc[:,"in_out"][data.loc[:, "in_out"] == "시외"] = 1
    data.loc[:,"in_out"][data.loc[:, "in_out"] == "시내"] = 0

    coords_jejusi = (33.500770, 126.522761) #제주시의 위도 경도
    coords_seoquipo = (33.259429, 126.558217) #서귀포시의 위도 경도

    data["dis_jejusi"] = (((data["latitude"] - coords_jejusi[0]) * 110000)**2 + ((data["longitude"] - coords_jejusi[1])* 88800)**2)**0.5
    data["dis_seoquipo"] = (((data["latitude"] - coords_seoquipo[0]) * 110000)**2 + ((data["longitude"] - coords_seoquipo[1])* 88800)**2)**0.5

    data["dis_jejusi"] /= 1000
    data["dis_seoquipo"] /= 1000

    train = data[input_var]
    if training:
        y = data["18~20_ride"]
        return train, y

    return train

def validation(data, model):

    val_data = data


if __name__ == "__main__":
    x, y = processing()
    print(x,y)




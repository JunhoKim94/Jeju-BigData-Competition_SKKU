import numpy as np
import pandas as pd
from torch import nn,tensor
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,MaxAbsScaler


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

def processing(path = "./data/train.csv", training = True, date_change = True, sort = "min_max",  input_var = ['in_out','latitude','longitude','6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride',
       '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff', '8~9_takeoff',
       '9~10_takeoff', '10~11_takeoff', '11~12_takeoff','weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6', 'dis_jejusi', 'dis_seoquipo']):

    '''
    id,date,bus_route_id,in_out,station_code,station_name,latitude,longitude,6~7_ride,7~8_ride,8~9_ride,9~10_ride,10~11_ride,11~12_ride,6~7_takeoff,7~8_takeoff,8~9_takeoff,9~10_takeoff,10~11_takeoff,11~12_takeoff
    '''


    data = pd.read_csv(path)

    #Data 변수 변환
    if date_change:
        data["date"] = pd.to_datetime(data["date"])
        data["weekday"] = data["date"].dt.weekday
        
        data = pd.get_dummies(data, columns = ["weekday"])

    #시외 = 1, 시내 = 0
    if "in_out" in input_var:
        data.loc[:,"in_out"][data.loc[:, "in_out"] == "시외"] = 1
        data.loc[:,"in_out"][data.loc[:, "in_out"] == "시내"] = 0

    if "dis_jejusi" in input_var:
        coords_jejusi = (33.500770, 126.522761) #제주시의 위도 경도
        data["dis_jejusi"] = (((data["latitude"] - coords_jejusi[0]) * 110000)**2 + ((data["longitude"] - coords_jejusi[1])* 88800)**2)**0.5
        data["dis_jejusi"] /= 1000
        
        data["dis_jejusi"] = scaler(data["dis_jejusi"], sort = sort)

    if "dis_seoquipo" in input_var:
        coords_seoquipo = (33.259429, 126.558217) #서귀포시의 위도 경도
        data["dis_seoquipo"] = (((data["latitude"] - coords_seoquipo[0]) * 110000)**2 + ((data["longitude"] - coords_seoquipo[1])* 88800)**2)**0.5
        data["dis_seoquipo"] /= 1000

        data["dis_seoquipo"] = scaler(data["dis_seoquipo"], sort = sort)

    train = data[input_var]

    if "bus_route_id" in input_var:

        train = pd.get_dummies(train,columns=['bus_route_id'])
        
        #del train["bus_route_id"]
    
    if "station_code" in input_var:

        train = pd.get_dummies(train, columns = ["station_code"])

        #del train["station_code"]

    
    if training:

        y = data["18~20_ride"]

        return train, y

    return train

def scaler(data, sort = "min_max"):
    '''
    Data Scaler
    inputs
    sort: min_max, robust, standard, max_obs
    data: 1D array or 2D array
    output
    scaled data(2D)
    '''
    if len(data.shape) == 1:
        data = data.reshape(-1,1)
    
    if sort.lower() == "min_max":
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)

    elif sort.lower() == "robust":
        scaler = RobustScaler()
        scaler.fit(data)
        data = scaler.transform(data)

    elif sort.lower() == "standard":
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

    elif sort.lower() == "max_abs":
        scaler = MaxAbsScaler()
        scaler.fit(data)
        data = scaler.transform(data)

    return data

if __name__ == "__main__":
    x, y = processing()
    print(x,y)




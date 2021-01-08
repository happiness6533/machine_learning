
import numpy as np
import pandas as pd

# 데이터 프레임
two_dimensional_list = [['dongwook', 50, 86], ['sineui', 89, 31], ['ikjoong', 68, 91], ['yoonsoo', 88, 75]]
my_df = pd.DataFrame(two_dimensional_list, columns=["name", "영어점수", "수학점수"], index=["a", "b", "c", "d"])
print(my_df.columns)
print(my_df.index)
print(my_df)
print(my_df.dtypes)

# 데이터 프레임 만들기1 : 배열을 담고 있는 배열로 + 2차원 넘파이 배열로 + 시리즈를 담고 있는 배열로
two_dimensional_list = [['dongwook', 50, 86], ['sineui', 89, 31], ['ikjoong', 68, 91], ['yoonsoo', 88, 75]]
two_dimensional_array = np.array(two_dimensional_list)
list_of_series = [
    pd.Series(['dongwook', 50, 86]),
    pd.Series(['sineui', 89, 31]),
    pd.Series(['ikjoong', 68, 91]),
    pd.Series(['yoonsoo', 88, 75])
]

# 아래 셋은 모두 동일합니다
df1 = pd.DataFrame(two_dimensional_list, columns=["name", "english_score", "math_score"])
df2 = pd.DataFrame(two_dimensional_array, columns=["name", "english_score", "math_score"])
df3 = pd.DataFrame(list_of_series, columns=["name", "english_score", "math_score"])


# 데이터 프레임2(딕셔너리로 만들기)
names = ['dongwook', 'sineui', 'ikjoong', 'yoonsoo']
english_scores = [50, 89, 68, 88]
math_scores = [86, 31, 91, 75]

dict1 = {
    'name': names,
    'english_score': english_scores,
    'math_score': math_scores
}

dict2 = {
    'name': np.array(names),
    'english_score': np.array(english_scores),
    'math_score': np.array(math_scores)
}

dict3 = {
    'name': pd.Series(names),
    'english_score': pd.Series(english_scores),
    'math_score': pd.Series(math_scores)
}

# 아래 셋은 모두 동일합니다
df4 = pd.DataFrame(dict1)
df5 = pd.DataFrame(dict2)
df6 = pd.DataFrame(dict3)



# 데이터 프레임3(사전이 들어있는 배열로 만들기)
my_list = [
    {'name': 'dongwook', 'english_score': 50, 'math_score': 86},
    {'name': 'sineui', 'english_score': 89, 'math_score': 31},
    {'name': 'ikjoong', 'english_score': 68, 'math_score': 91},
    {'name': 'yoonsoo', 'english_score': 88, 'math_score': 75}
]

df7 = pd.DataFrame(my_list)
print(df7)



# 다양한 데이터 타입이 존재
# object : 텍스트
# datatime64 : 날짜와 시간
# category : 카테고리
print(df1.dtypes)








# csv 파일 : 컴마로 구분되어 있는 데이터, 첫 줄은 반드시 헤더, 만약 헤더가 없는 경우에는 read_csv('경로', header=None)
iphone_df1 = pd.read_csv('data/iphone.csv', index_col=0)  # 0번 칼럼을 인덱스, 즉 각 row의 이름으로 설정한다
print(iphone_df1)

# iphone_df2 = pd.read_csv('data/iphone.csv', index_col="출시일")  # 이름을 직접 설정해도 된다
# print(iphone_df2)


# csv 인덱싱 : 여러줄을 동시에 선택하려면 배열로 선택한다 : loc[row, column]
# 그런데, loc를 쓰지 않아도 작동한다
print(iphone_df1.loc["iPhone 7"])

#
#
#
# print(iphone_df1.loc["iPhone 7", "출시일"])
# print(iphone_df1.loc["iPhone 7", ["출시일", "디스플레이"]])
# print(iphone_df1.loc["iPhone 7", :])
# print(iphone_df1.loc[:, "출시일"])
# # loc를 쓰지 않고도 인덱싱을 할 수 있다 : print(iphone_df["출시일"])
# print(iphone_df1.loc["iphone 7":"iphone X", :])
# print(iphone_df1.loc[:, "메모리":"Face ID"])

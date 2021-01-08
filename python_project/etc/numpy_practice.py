import numpy as np

# 넘파이 배열1
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr1)
print(arr1.shape)
print(arr1.size)

# 넘파이 배열2
arr2 = np.arange(0, 10)
arr3 = np.arange(10, 20)
arr4 = np.arange(0, 10, 2)

#  넘파이 배열 인덱스, 슬라이싱
print(arr2[0])
print(arr2[-1])
print(arr2[[0, 1, -1]])
print(arr2[1:9])
print(arr2[1:])
print(arr2[:9])
print(arr2[1:10:2])

# 넘파이 배열 불린1
print(arr2 > 5)
print(arr2 % 2 == 0)

# 넘파이 배열 불린2
arr_ex = np.array([1, 2, 3, 4, 5])
bools = (arr_ex <= 3)
index_filter = np.where(bools) # True에 해당하는 인덱스만 받아온다
print(arr_ex[index_filter])

# 넘파이 배열 불린3
print(arr_ex[arr_ex <= 3])


# 넘파이 배열 필터
filter = np.where(arr2 > 5)
filterd_arr = arr2[filter]
print(filterd_arr)

# 넘파이 배열 최대, 최소, 평균, 중앙값, 분산, 표준편차
print(arr2.max())
print(arr2.min())
print(arr2.mean())
print(np.median(arr2))
print(arr2.var())
print(arr2.std())

# 넘파이 배열 연산
arr22 = arr2 * 2
arr23 = arr2 + arr3
print(arr22)
print(arr23)

arr5 = np.full(6, 0)
arr6 = np.zeros(6, dtype=int)
arr7 = np.ones(6, dtype=int)
arr8 = np.random.random(6)
print(arr5)
print(arr6)
print(arr7)
print(arr8)

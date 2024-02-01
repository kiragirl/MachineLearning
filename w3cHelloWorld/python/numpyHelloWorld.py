import numpy as np

print(np.__version__)

arr = np.array([1, 2, 3, 4, 5])
print(type(arr))
print(arr)

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print('5th element on 2nd dim: ', arr[1, 4])
print("------------------数组迭代------------------------")
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
    for y in x:
        for z in y:
            print(z)
print("------------------使用 nditer() 迭代数组------------------------")

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr):
    print(x)
print("------------------迭代时更改元素的数据类型------------------------")
# 迭代时更改元素的数据类型
arr = np.array([1, 2, 3])

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
    print(x)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("------------------以不同的步长迭代------------------------")
for x in np.nditer(arr[:, ::2]):
    print(x)
print("------------------使用 ndenumerate() 进行枚举迭代------------------------")
arr = np.array([1, 2, 3])

for idx, x in np.ndenumerate(arr):
    print(idx, x)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(arr):
    print(idx, x)
print("------------------连接 NumPy 数组------------------------")
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))

print(arr)
print("------------------连接 NumPy 数组------------------------")
arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)

print(arr)
print("------------------使用堆栈函数连接数组------------------------")
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)
print("------------------沿行堆叠------------------------")
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.hstack((arr1, arr2))

print(arr)
print("------------------沿列堆叠------------------------")
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.vstack((arr1, arr2))

print(arr)
print("------------------沿高度堆叠（深度）------------------------")
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.dstack((arr1, arr2))

print(arr)
print("------------------NumPy 数组拆分------------------------")
arr = np.array([1, 2, 3, 4, 5, 6, 7])

newarr = np.array_split(arr, 3)

print(newarr)

print("------------------分割二维数组------------------------")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3)

print(newarr)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3, axis=1)

print(newarr)
print("------------------搜索数组------------------------")
arr = np.array([1, 2, 3, 4, 5, 4, 4])

x = np.where(arr == 4)

print(x)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 1)

print(x)
print("------------------搜索有序数组------------------------")
arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7)

print(x)

arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7, side='right')

print(x)
print("------------------查找应在其中插入值 2、4 和 6 的索引------------------------")

arr = np.array([1, 3, 5, 7])

x = np.searchsorted(arr, [2, 4, 6])

print(x)
print("------------------数组排序------------------------")
arr = np.array([[3, 2, 4], [5, 0, 1]])

print(np.sort(arr))
print("------------------布尔索引列表来过滤数组------------------------")

arr = np.array([61, 62, 63, 64, 65])

x = [True, False, True, False, True]

newarr = arr[x]

print(newarr)

print("------------------创建过滤器数组------------------------")

arr = np.array([61, 62, 63, 64, 65])

# 创建一个空列表
filter_arr = []

# 遍历 arr 中的每个元素
for element in arr:
  # 如果元素大于 62，则将值设置为 True，否则为 False：
  if element > 62:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)
print("------------------直接从数组创建过滤器------------------------")
arr = np.array([61, 62, 63, 64, 65])

filter_arr = arr > 62
print(filter_arr)
newarr = arr[filter_arr]
print(newarr)
print("------------------通用函数------------------------")
print("------------------通用函数------------------------")
x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = np.add(x, y)

print(z)
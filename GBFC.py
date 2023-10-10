import numpy as np
from sklearn import preprocessing
from scipy.optimize import minimize


def process_dataset(file_path):  # 数据集读取
    # 读取数据集文件 delimiter表示用什么隔开，默认是空格
    data = np.loadtxt(file_path, delimiter=',')
    # 随机打乱数据集
    np.random.shuffle(data)
    # 将数据集分为特征和标签
    features = data[:, :-1]
    labels = data[:, -1]

    return features, labels


def convert_array_to_matrix(arr, n):  # 此方法用来把读取的二维数组转换为算法需要的dxn矩阵形式，其中n是样本数，d为特征维度
    # 转置数组并返回
    return np.transpose(arr).reshape(4, n)


def standardscaler(R):  # 标准化处理方法
    # 创建一个StandardScaler对象，用于进行标准化处理
    scaler = preprocessing.StandardScaler()

    # 使用fit_transform方法进行标准化处理
    normalized_data = scaler.fit_transform(R)
    return normalized_data


def generate_matrix(n, m):
    F = np.random.rand(n, m)  # 生成一个n x m的随机矩阵，元素取值范围在[0,1)
    # 将每行的和归一化为1
    F = F / np.sum(F, axis=1, keepdims=True)

    return F


def multi(A, B):  # 矩阵乘法
    return np.dot(A, B)


def tran(A):  # 矩阵转置
    return np.transpose(A)


def div(A, B):  # 矩阵除法
    return np.linalg.solve(B, A)


def sqrt(A):  # 矩阵开方
    return np.sqrt(A)


# 这下面4个方法都是为了求解拉格朗日函数和KKT条件的
def objective(F1_F, F):
    F1_F = F1_F.reshape(F.shape)
    F1 = F1_F[:F.shape[0], :]
    return np.linalg.norm(F1 - F, ord='fro') ** 2


def constraint1(F1):
    return np.sum(F1, axis=1) - 1


def constraint2(F1):
    return np.minimum(F1, 0)


def minimize_f1(F):
    n, m = F.shape
    initial_guess = np.random.rand(n * m)

    bounds = [(0, None)] * (n * m)
    constraints = [{'type': 'eq', 'fun': constraint1}, {'type': 'ineq', 'fun': constraint2}]

    result = minimize(objective, initial_guess, args=(F,), bounds=bounds, constraints=constraints)
    F1_F = result.x.reshape(F.shape)
    F1 = F1_F[:F.shape[0], :]

    return F1
def update(L, t, a, f, n, c):  # L是算出来的拉普拉斯矩阵，t是超参数，a是梯度下降算法中的步长, f是外面传进来的一个空数组，满足每行和为1，且元素非负
    f_ = f
    tmp = np.random.rand(n, c)
    for i in range(0,100):  # 迭代次数
        f = f_ - a * (2 * multi(L, f_) - t * div(f_, sqrt(multi(tran(f_), f_))))  # 通过梯度下降算法获得下一层的f
        # 接下来要通过拉普拉斯函数和KTT条件来优化出新的f_
        f_ = minimize_f1(f)
        if tmp == f_:
            break
        tmp = f_  # 用来检查是否收敛

    return f_  # 这就是最后的隶属度矩阵，也就是最终答案了







features, labels = process_dataset("C:\\Users\\Celes\\Desktop\\iris\\data.txt")
features = standardscaler(features)  # 对样本特征进行标准化处理 (这一步我不知道可不可以，待定）
# 准备一些后面要用的变量
n = len(features)
c = len(features[0])
f = generate_matrix(n, c)

features_matrix = convert_array_to_matrix(features, len(features))  # 获得特征值的矩阵 dxn规模



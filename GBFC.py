'''
写在前面：这个算法有几个地方是可以调整的，学习率，步长，迭代次数，锚定集数量的选取，邻居数量的选取，很多地方可以改进
'''
import numpy as np
from sklearn import preprocessing
from scipy.optimize import minimize
import math
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def process_dataset(file_path):  # 数据集读取
    # 读取数据集文件 delimiter表示用什么隔开，默认是空格
    data = np.loadtxt(file_path, delimiter=',')
    # 随机打乱数据集
    np.random.shuffle(data)
    # 将数据集分为特征和标签
    features = data[:, :-1]
    labels = data[:, -1]

    return features, labels


def convert_array_to_matrix(arr):  # 此方法用来把读取的二维数组转换为算法需要的dxn矩阵形式，其中n是样本数，d为特征维度
    # 转置数组并返回
    return np.transpose(arr)


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


def div(A, B):
    A = convert_array_to_matrix(A)
    B = convert_array_to_matrix(B)
    C = convert_array_to_matrix(inverse(B))
    return multi(A, C)


def sqrt(A):  # 矩阵开方
    return np.sqrt(A)


# 这下面4个方法都是为了求解拉格朗日函数和KKT条件的
def objective(x, f):
    f1 = x.reshape(f.shape)
    return np.linalg.norm(f1 - f)  # 求解F范数最小化的目标函数

def constraint(x):
    return np.sum(x) - 1  # 约束条件：f1每行元素之和等于1

def minimize_f(f):
    m, n = f.shape
    x0 = np.random.rand(m * n)  # 随机初始化f1的取值
    # 定义约束和目标函数
    con = {'type': 'eq', 'fun': constraint}
    bounds = [(0, None) for _ in range(m * n)]  # 每个变量的取值范围（大于等于0）

    # 求解优化问题
    res = minimize(objective, x0, args=(f,), constraints=con, bounds=bounds)
    f1 = res.x.reshape(f.shape)

    return f1



def anchor_set(samples, m):
    distances = []
    for sample in samples:
        distance = math.sqrt(sum([x**2 for x in sample]))  # 计算样本的欧几里得距离
        distances.append(distance)
    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k], reverse=True)
    farthest_samples = [samples[i] for i in sorted_indices[:m]]
    return farthest_samples


def sub_columns(A, B, a, b):  # 用来算矩阵的向量减法 a b为向量号 向量号是从0开始！
    A = np.array(A)
    B = np.array(B)
    column_a = A[:, a]
    column_b = B[:, b]
    result = column_a - column_b
    return result.tolist()


def calculate_l2_norm(array):  # 计算L2范数
    array = np.array(array)
    l2_norm = np.linalg.norm(array)
    return l2_norm


def create_Z(x, u, n, m, k):  #用来求z
    z = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            if j <= k:
                tmp = 0
                for t in range(0, k):
                    tmp += calculate_l2_norm(sub_columns(x, u, i, t))
                z[i][j] = ((calculate_l2_norm(sub_columns(x, u, i, k + 1))) - (
                    calculate_l2_norm(sub_columns(x, u, i, j)))) / (
                                      (k * (calculate_l2_norm(sub_columns(x, u, i, k + 1)))) - tmp)

            else:
                z[i][j] = 0

    return z


def inverse(matrix):  # 求矩阵转置
    # 使用numpy库将矩阵转换为numpy数组
    matrix_array = np.array(matrix)

    # 使用numpy库的线性代数模块计算逆矩阵
    try:
        inverse_matrix_array = np.linalg.inv(matrix_array)
        return convert_array_to_matrix(inverse_matrix_array.tolist())
    except np.linalg.LinAlgError:  # 防止矩阵不可逆
        return "矩阵不可逆"



def create_tri(z, m):  # 用来求三角形
    tri = np.zeros((m, m))
    tmp = 0
    for i in range(0, m):
        for j in range(0, m):
            tmp += z[i][j]
        tri[i][i] = tmp
    tri = np.transpose(tri).reshape(m, m)  # 把tri转换为矩阵形式
    return tri


def update(L, t, a, f, n, c):  # L是算出来的拉普拉斯矩阵，t是超参数，a是梯度下降算法中的步长, f是外面传进来的一个空数组，满足每行和为1，且元素非负
    f_ = f
    tmp = np.zeros((len(f_),len(f_[0])))
    tmp = convert_array_to_matrix(tmp)
    for i in range(0,100):  # 迭代次数
        A = 2*multi(L,f_)
        B = t*f_
        C = sqrt(multi(tran(f_),f_))
        T = convert_array_to_matrix(inverse(C))
        D = multi(B,T)
        f = f_ - a*(A-D)
        # 接下来要通过拉普拉斯函数和KTT条件来优化出新的f_
        f_ = minimize_f(f)
        if np.array_equal(tmp,f_):
            break
        tmp = f_  # 用来检查是否收敛

    return f_  # 这就是最后的隶属度矩阵，也就是最终答案了


def normalized_entropy(predicted_labels, true_labels):
    # 对预测标签和真实标签进行编码
    label_encoder = LabelEncoder()
    label_encoder.fit(true_labels)
    predicted_labels_encoded = label_encoder.transform(predicted_labels)
    true_labels_encoded = label_encoder.transform(true_labels)

    # 统计每个聚类簇的标签频率
    cluster_counts = Counter(predicted_labels_encoded)
    unique_true_labels = np.unique(true_labels_encoded)

    # 计算每个聚类簇的熵值
    entropies = []
    for cluster_label in cluster_counts.keys():
        cluster_indices = (predicted_labels_encoded == cluster_label)
        cluster_true_labels = true_labels_encoded[cluster_indices]
        cluster_true_label_counts = Counter(cluster_true_labels)

        cluster_entropy = 0
        for true_label in unique_true_labels:
            true_label_count = cluster_true_label_counts.get(true_label, 0)
            if true_label_count > 0:
                p = true_label_count / len(cluster_true_labels)
                cluster_entropy -= p * np.log2(p)

        entropies.append(cluster_entropy)
        # 计算总样本数
        total_samples = len(true_labels)

        # 计算归一化熵值
        normalized_entropy = np.sum([e * c for e, c in zip(entropies, cluster_counts.values())]) / total_samples

        # 将归一化熵值减去1，并取绝对值得到NE指标
        ne = abs(normalized_entropy - 1)

        return ne


# 目前缺少的是求A的手法，关键在于knn使用，以及锚定集的选取。
# 锚定集的选取我有了一个想法，取样本中欧氏距离最远的m个样本，设定为锚定集


features, labels = process_dataset("C:\\Users\\Celes\\Desktop\\iris\\data.txt")
features = standardscaler(features)  # 对样本特征进行标准化处理 (这一步我不知道可不可以，待定）
u = anchor_set(features, 9)  # 这里取锚定集中的元素个数为9
# 准备一些后面要用的变量
n = len(features)
c = len(features[0])
f = generate_matrix(n, 3)
F = convert_array_to_matrix(features)  # 获得特征值的矩阵 dxn规模  即X
U = convert_array_to_matrix(u)  # 即U 锚定集 dxm规模
k = 5  # k是一行中不为0的元素个数
m = 9

Z = create_Z(F,U,n,m,k)  # 这里我们得到了Z
tri = create_tri(Z,m)
tri_inverse = inverse(tri)
tmp = multi(Z,tri_inverse)
A = multi(tmp,tran(Z))  # 这里就求出了A

D = np.zeros((n,n))
for i in range(0,n):
    tmp_d = 0
    for j in range(0,n):
        tmp_d += A[i][j]
    D[i][i] = tmp_d
L = D - A  # 这里就已经求出了L，即拉普拉斯矩阵
f_ = update(L,0.5,0.5,f,n,c)
cluster = [-1] * len(f_)
tmp_c = -1
for i in range(len(f_)):  # 求出聚类的结果，并保存在cluster中
    max = 0
    for j in range(len(f_[0])):
        if f_[i][j] > max:
            max = f_[i][j]
            tmp_c = j
    cluster[i] = tmp_c
print(cluster)
count = [0] * 3
for i in cluster:
    if i == 0:
        count[0] += 1
    elif i == 1:
        count[1] += 1
    else:
        count[2] += 1

print("********************************************************************")
print(count)
ne = normalized_entropy(cluster,labels)
print("ne:",ne)

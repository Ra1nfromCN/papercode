'''
输入是客户集C, 每个簇的规模上下限Bl,Bu,要分成几个簇k，还有一个实数e属于（0,1].
输出R,抽样的结果。
'''
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import adjusted_rand_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter


# 得到的数据集是二维数组的形式 数据集预处理
def process_dataset(file_path):
    # 读取数据集文件 delimiter表示用什么隔开，默认是空格
    data = np.loadtxt(file_path, delimiter=',')
    # 随机打乱数据集
    np.random.shuffle(data)
    # 将数据集分为特征和标签
    features = data[:, :-1]
    labels = data[:, -1]

    return features, labels


# 计算样本间的距离
def calculate_distance(f, row1, row2):
    # 获取对应行的特征数据
    x1 = features[row1]
    x2 = features[row2]

    # 计算欧几里得距离
    distance = np.sqrt(np.sum((x1 - x2) ** 2))

    return distance


def create_array():
    # 创建一个长度为3的一维数组，初始值为-1
    arr = [-1] * 3
    return arr


def create_c(length):
    # 创建一个一维数组，长度为length
    arr = np.arange(length)

    return arr


def modify_array(arr, num):
    # 遍历数组
    for i in range(len(arr)):
        # 如果数组中的元素等于-1
        if arr[i] == -1:
            # 将该元素修改为传入的整数
            arr[i] = num
            # 修改完毕后直接返回数组
            return np.array(arr)
    # 如果数组中没有值为-1的元素，则返回原数组
    return np.array(arr)


def notEmpty_array(arr):
    # 遍历数组 判空
    for i in range(len(arr)):
        if arr[i] != -1:
            return True
    return False


def isFull_array(arr):
    for i in range(len(arr)):
        if arr[i] == -1:
            return False
    return True


def find_farthest_sample(c, R, f):
    # 随机抽取一半的样本
    num_samples = len(c) // 2
    if(num_samples==0):
        num_samples = 1
    sampled_c = np.random.choice(c, size=num_samples, replace=False)

    # 计算每个样本与R中对应索引的样本距离的总和
    distances = []
    for sample in sampled_c:
        sample_distance = np.sum([calculate_distance(f, sample, r) for r in R if r != -1])  # 这里r != -1 防止了空簇的出现
        distances.append(sample_distance)  # 把距离存进去
    if len(distances) == 0:
        return sampled_c[0]
    # 找到距离总和最远的样本的索引
    farthest_sample_index = np.argmax(distances)

    # 创建一个一维数组，存放距离总和最远的样本
    farthest_sample = np.array([sampled_c[farthest_sample_index]])

    return farthest_sample


# 对数据进行标准化处理
def standardscaler(R):
    # 创建一个StandardScaler对象，用于进行标准化处理
    scaler = preprocessing.StandardScaler()

    # 使用fit_transform方法进行标准化处理
    normalized_data = scaler.fit_transform(R)
    return normalized_data


# 上面的为下面的主要算法服务
# *******************************************************************************************************************************
# *******************************************************************************************************************************
# *******************************************************************************************************************************
# *******************************************************************************************************************************
# 三个主要的函数组成了这篇论文的算法

def sampling(c, e, R, k, f):  # R是空集。e是超参数，自己定的，大于0，最大可取到1.
    tmp = int(float(k)/e)
    for i in range(0,tmp):
        Rc = create_array()
        partitioning(k, Rc, c, R, f)
        # Rc是一次获得的客户集，一开始是空的，R里面有很多个客户集，是每次取样取到的。
    return R


def partitioning(k, Rc, c, R, f):  # R是保存最终结果的集合(set)  c是一维数组，存放的都是索引 ，features单独传, Rc就还是一维数组
    if(isFull_array(Rc)):
        R = R.add(tuple(Rc))
    else:
        if(len(c)>1):
            random_int = np.random.randint(0, len(c) - 1)
        else:
            random_int = 0
        num = c[random_int]
        # 随机抽取一个c
        partitioning(k, modify_array(Rc, num), c, R, f)  # 传入修改后的数组进行递归
        if notEmpty_array(Rc):
            partitioning(k, Rc, find_farthest_sample(c, Rc, f), R, f)


def selectioning(k, R, f):  # k是簇的数量，R是算法1抽样得到的,以二维数组的形式出现，f是features，也是二维数组
    R = np.array(list(R))  # 把R从集合转换为二维数组
    U = np.empty((len(R)*(k**k), k+1))  # U是用来保存答案的
    tick = 0  # 下面会用到
    for a in range(0, len(R)):  # 每一组都要用
        # 这个循环是代表要做这么多次
        for b in range(0, k ** k):
            code = [-1] * len(f)
            # 给f中每一个样本随机编号
            for i in range(0, len(f)):
                code[i] = np.random.randint(0, k)
            H = [-1] * (k+1)
            for t in range(0, k):
                distance = 9999999  # 算距离
                flag = -1  # 记录索引
                # 要找编号一样的(算法的要求）
                for c in range(0, len(code)):
                    if code[c] == t:
                        Rtmp = R[a][t]
                        tmp = calculate_distance(f, c, Rtmp)
                        if tmp < distance:
                            distance = tmp
                            flag = c
                H[t] = flag  # 捕获最小的ht的索引 即得到聚类中心
            # 这里开始计算花费 此时我们有的是H数组，前三格存放的是三个聚类中心的索引，第四格是待定的，用来保存后面算出来的代价
            # 我想要做的是 我想统计聚类的结果 并且输出准确率
            cluster = [-1] * len(f)  # 用来保存聚类的结果
            total = 0  # 计算花费
            count = [0] * k  # 用来保存看这一种聚类方法的分配情况，并施加平衡约束
            for i in range(0, len(f)):  # 求每一个样本，在这次循环中属于哪个聚类中心
                min = 9999
                flag = -1
                for t in range(0, k):
                    if H[t] == i:
                        cluster[i] = i  # 聚类中心肯定是属于自己那一类的
                        flag = t
                    else:
                        if count[t] >= ((len(f)//k)+5):  # 限制上限 即Bu 而且此时Bu的限制可以不需要设置Bl
                            continue  # 直接不去算这距离
                        d = calculate_distance(f, H[t], i)  # 计算欧氏距离
                        if d < min:
                            min = d
                            cluster[i] = H[t]
                            flag = t  # 用来保存这个样本存入的聚类中心
                            # 找到属于哪一类
                if flag != -1:
                    count[flag] += 1
                total += min  # 计算总消费
            H[-1] = total
            # 这时候就获得了一组H了，包含k个聚类中心和这种聚类方法的连接花费
            U[tick] = H  # tick用来把H按顺序存入U中
            tick += 1
    return U


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


features, labels = process_dataset("C:\\Users\\Celes\\Desktop\\iris\\data.txt")
features = standardscaler(features)  # 对数据进行标准化处理
f = features  # 特征 二维数组
c = create_c(len(f))  # c是索引  方便数组使用的
Re = set()  # Re一开始是空的
k = 3  # 要分几类
e = 0.5  # 超参数 决定了迭代次数吧
R = sampling(c, e, Re, k, f)
U = selectioning(k, R, f)  # U是求出来的所有聚类中心和花费

# 遍历U 找出最小值 即最终选出的facilities，以及它的connection function
min = 9999
flag = -1

for i in range(0,len(U)):
    if U[i][-1] < min:
        min = U[i][-1]
        flag = i

KEY = U[flag]  # KEY代表聚类中心和花费
tmp = -1
cluster = [-1] * len(f)  # cluster用来保存聚类结果
count = [0] * k  # 计算每个中心连接的数量

for i in range(0, len(f)):
    min = 9999
    flag = -1
    for t in range(0, k):  # 这里的求解方法是 计算这个样本与这三个聚类中心的距离，选择最近的那个作为自己连接的对象
        tmp = int(KEY[t])
        if tmp == i:
            cluster[i] = i  # 聚类中心肯定是属于自己那一类的
            flag = t
        else:
            if count[t] >= ((len(f)//k)+5):
                continue
            d = calculate_distance(f, tmp, i)  # 计算欧氏距离
            if d < min:
                min = d
                cluster[i] = tmp
                flag = t
    if flag != -1:
        count[flag] += 1

print("聚类中心和连接花费为", KEY)
print("具体的连接情况为", cluster)
ne_label = [0] * len(f)
t = 0
for i in cluster:  # 配置计算NE用的预测标签数组
    if i == int(KEY[0]):
        ne_label[t] = 0
        t += 1
    elif i == int(KEY[1]):
        ne_label[t] = 1
        t += 1
    elif i == int(KEY[2]):
        ne_label[t] = 2
        t += 1

print("每种类型样本的数量为",count)  # 计算每个簇的样本数量
# 计算ARI
ari = adjusted_rand_score(labels, cluster)
print("Adjusted Rand Index (ARI):", ari)
ne = normalized_entropy(ne_label, labels)
print(ne_label)
print("normalized entropy (NE):", ne)

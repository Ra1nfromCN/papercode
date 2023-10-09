'''
输入是客户集C, 每个簇的规模上下限Bl,Bu,要分成几个簇k，还有一个实数e属于（0,1].
输出R,抽样的结果。
'''
import numpy as np
from sklearn import preprocessing

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
        sample_distance = np.sum([calculate_distance(f, sample, r) for r in R if r != -1])
        distances.append(sample_distance)
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



def Sampling(c,e,R,k,f):
# R是空集。e是超参数，自己定的，大于0，最大可取到1.
    tmp = int(float(k)/e)
    for i in range(0,tmp):
        Rc = create_array()
        Partitioning(k,Rc,c,R,f)
        # Rc是一次获得的客户集，一开始是空的，R里面有很多个客户集，是每次取样取到的。
    return R


def Partitioning(k,Rc,c,R,f): # R是保存最终结果的集合(set)  c是一维数组，存放的都是索引 ，features单独传, Rc就还是一维数组
    if(isFull_array(Rc)):
        R = R.add(tuple(Rc))
    else:
        if(len(c)>1):
            random_int = np.random.randint(0, len(c) - 1)
        else:
            random_int = 0
        num = c[random_int]
        # 随机抽取一个c
        Partitioning(k,modify_array(Rc,num),c,R,f) # 传入修改后的数组进行递归
        if(notEmpty_array(Rc)):
            Partitioning(k,Rc,find_farthest_sample(c,Rc,f),R,f)

def Selectioning(k,R,f):  #  k是簇的数量，R是算法1抽样得到的,以二维数组的形式出现，f是features，也是二维数组
    R = np.array(list(R))  # 把R从集合转换为二维数组
    U = np.empty((len(R)*(k**k),k+1))
    count = 0
    for a in range(0,len(R)):  # 每一组都要用
        # 这个循环是代表要做这么多次
        for b in range(0, k ** k):
            code = [-1] * len(f)
            # 给f中每一个样本随机编号
            for i in range(0, len(f)):
                code[i] = np.random.randint(0, k)
            H = [-1] * (k+1)
            for t in range(0, k):
                distance = 9999999 # 算距离
                flag = -1  # 记录索引
                # 要找编号一样的
                for c in range(0, len(code)):
                    if code[c] == t:
                        Rtmp = R[a][t]
                        tmp = calculate_distance(f,c,Rtmp)
                        if(tmp < distance):
                            distance = tmp
                            flag = c
                H[t] = flag  # 捕获最小的ht的索引 即得到聚类中心
            # 这里开始计算花费 此时我们有的是H数组，前三格存放的是三个聚类中心的索引，第四格是待定的，用来保存后面算出来的代价
            # 我想要做的是 我想统计聚类的结果 并且输出准确率
            cluster = [-1] * len(f)  # 用来保存聚类的结果
            spend = [-1] * k
            min = 9999
            total = 0
            for i in range(0,len(f)):  # 求每一个样本，在这次循环中属于哪个聚类中心
                for t in range(0,k):
                    if H[t] == i:
                        cluster[i] = i  # 聚类中心肯定是属于自己那一类的
                    else:
                        d = calculate_distance(f, H[t], i)  # 计算欧氏距离
                        if d < min:
                            min = d
                            cluster[i] = H[t]
                            # 找到属于哪一类
                total += min  # 计算总消费
                min = 9999
            H[-1] = total
            # 这时候就获得了一组H了，包含聚类中心，H的末尾是这种聚类方法的连接花费
            U[count] = H
            count += 1
    return U


features, labels = process_dataset("C:\\Users\\Celes\\Desktop\\iris\\data.txt")
features = standardscaler(features)  # 对数据进行标准化处理
f = features
c = create_c(len(f))  # c是索引
Re = set() # Re一开始是空的
k = 3
e = 0.5
R = Sampling(c,e,Re,k,f)
U = Selectioning(k,R,f)

# 找出最小值 即最终选出的facilities，以及它的connection function
min = 9999
flag = -1

print(U)
for i in range(0,len(U)):
    if U[i][-1] < min:
        min = U[i][-1]
        flag = i
print("******************")

KEY = U[flag]  # KEY代表聚类中心和花费
tmp = -1
cluster = [-1] * len(f)  # cluster用来保存聚类结果
count = [0] * k
for i in range(0, len(f)):
    min = 9999
    for t in range(0, k):
        tmp = int(KEY[t])
        if tmp == i:
            cluster[i] = i  # 聚类中心肯定是属于自己那一类的

        else:
            d = calculate_distance(f, tmp, i)  # 计算欧氏距离
            if d < min:
                min = d
                cluster[i] = tmp
print(KEY)
print(len(cluster))
for i in cluster:
    print(i)
    if i == int(KEY[0]):
        count[0] += 1
    elif i == int(KEY[1]):
        count[1] += 1
    elif i == int(KEY[2]):
        count[2] += 1

print(count)

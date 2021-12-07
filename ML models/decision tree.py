import numpy as np
from collections import Counter
from sklearn import datasets
import xlrd
import datetime

s1=datetime.datetime.now()
class DecisionTree:
    def __init__(self, algorithm='ID3'):
        """选择谁用的算法,可选的有ID3,C4.5,CARTcla(CART分类树),CARTreg(CART回归树)"""
        self.algorithm = algorithm

    @staticmethod
    def cal_entroy(dataset):
        """
        计算数据集的经验熵,数据集为np.array
        :param dataset: 数据集m*n,m为样本数,n为特征数
        :return: 数据集的经验熵
        """
        m = dataset.shape[0]  # 样本数
        labels = Counter(dataset[:, -1].reshape(m).tolist())  # 获取类别及其出现的次数
        entroy = 0  # 初始化经验熵
        for amount in labels.values():
            prob = amount / m  # 计算概率pi
            entroy -= prob * np.log(prob)  # e=-sum(pi*log(pi))
        return entroy

    @staticmethod
    def cal_gini(dataset):
        """
        计算数据集的基尼指数,数据集为np.array
        :param dataset: 数据集m*n,m为样本数,n为特征数
        :return: 数据集的基尼指数
        """
        m = dataset.shape[0]
        labels = Counter(dataset[:, -1].reshape(m).tolist())
        gini = 1
        for amount in labels.values():
            prob = amount / m
            gini -= prob ** 2  # g=1-sum(pi**2)
        return gini

    @staticmethod
    def cal_se(dataset):
        """
        计算数据集的方差squared error,数据集为np.array
        np.var可直接计算出均方差,乘以样本数即为方差
        :param dataset: 数据集m*n,m为样本数,n为特征数
        :return: 数据集的方差
        """
        return np.var(dataset[:, -1]) * dataset.shape[0] if dataset.shape[0] > 0 else 0

    def split_dataset(self, dataset, feature, value):
        """
        根据特征feature的特征值value,划分数据集
        :param dataset: 数据集m*(n+1),m为样本数,n为特征数
        :param feature: 作为划分点的特征的索引
        :param value: 特征的某一个值
        :return: dataset[feature]==value的数据集,且不再包含feature特征
        """
        m, n = dataset.shape[0], dataset.shape[1] - 1
        if self.algorithm == 'ID3' or self.algorithm == 'C4.5':  # 获取所有特征值等于给定值的样本D,返回去掉该特征列的D.
            split_data = np.zeros((1, n))  # 初始化一个1*n的二维数组,便于使用np.concatenate来增添数据,最后输出结果时再去掉第一行就OK.
            for i in range(m):
                if dataset[i, feature] == value:
                    temp = np.concatenate((dataset[i, : feature], dataset[i, feature + 1:])).reshape(1, n)
                    split_data = np.concatenate((split_data, temp))
            return split_data[1:, :]
        else:  # 获取符合条件的样本,用于CART
            if self.algorithm == 'CARTcla':  # CART分类树,训练数据为离散型
                left = dataset[np.nonzero(dataset[:, feature] == value)[0], :]
                right = dataset[np.nonzero(dataset[:, feature] != value)[0], :]
            else:  # CART回归树,训练数据为连续型
                left = dataset[np.nonzero(dataset[:, feature] <= value)[0], :]
                right = dataset[np.nonzero(dataset[:, feature] > value)[0], :]
            return left, right

    def cal_entroy_gain(self, base_ent, dataset, feature):
        """
        计算信息增益,用于ID3
        :param base_ent: 原数据的经验熵
        :param dataset: 数据集m*(n+1),m为样本数,n为特征数
        :param feature: 作为划分点的特征的索引
        :return: 按照指定特征划分后的信息增益
        """
        new_ent = 0
        values = np.unique(dataset[:, feature])  # 获取特征值的取值范围
        for value in values:
            new_ent += self.cal_entroy(self.split_dataset(dataset, feature, value))
        return base_ent - new_ent

    def cal_entroy_gain_rate(self, base_ent, dataset, feature):
        """
        计算信息增益比,用于C4.5
        :param base_ent: 原数据的经验熵
        :param dataset: 数据集m*(n+1),m为样本数,n为特征数
        :param feature: 作为划分点的特征的索引
        :return: 按照指定特征划分后的信息增益比
        """
        new_ent, split_ent = 0, 0
        values = np.unique(dataset[:, feature])
        for value in values:
            split_data = self.split_dataset(dataset, feature, value)
            new_ent += self.cal_entroy(split_data)
            prob = split_data.shape[0] / dataset.shape[0]
            split_ent -= prob * np.log(prob)
        return (base_ent - new_ent) / split_ent

    def cal_split_gini(self, dataset, feature):
        """
        计算数据集按照某一特征的值划分后,可以取得的最小基尼指数,返回该基尼指数和对应的值. 用于CART分类树
        :param dataset: 数据集m*(n+1),m为样本数,n为特征数
        :param feature: 作为划分点的特征的索引
        :return: 最小基尼指数与其对应的特征值
        """
        values = np.unique(dataset[:, feature])
        min_gini, min_value = np.inf, 0
        for value in values:
            left, right = self.split_dataset(dataset, feature, value)
            new_gini = left.shape[0] / dataset.shape[0] * self.cal_gini(left) + right.shape[0] / dataset.shape[0] * \
                       self.cal_gini(right)
            if new_gini < min_gini:
                min_gini = new_gini
                min_value = value
        return min_gini, min_value

    def cal_split_se(self, dataset, feature):
        """
        计算数据集按照某一特征的值划分后,可以取得的最小方差,返回该方差和对应的值. 用于CART回归树
        :param dataset: 数据集m*(n+1),m为样本数,n为特征数
        :param feature: 作为划分点的特征的索引
        :return: 最小基尼指数与其对应的特征值
        """
        values = np.unique(dataset[:, feature])
        min_se, min_value = np.inf, 0
        for value in values:
            left, right = self.split_dataset(dataset, feature, value)
            new_se = self.cal_se(left) + self.cal_se(right)
            if new_se < min_se:
                min_se = new_se
                min_value = value
        return min_se, min_value

    def choose_best_feature(self, dataset):
        """
        根据各算法的要求,选取对划分数据效果最好的特征.
        :param dataset: 数据集m*(n+1),m为样本数,n为特征数
        :return: 对于ID3和C.45,返回最佳特征的索引值;对于CART回归树和分类树,返回最佳特征的索引值和对应的特征值
        """
        m, n = dataset.shape[0], dataset.shape[1] - 1
        base_ent = self.cal_entroy(dataset)
        delta_gini, delta_info = np.inf, -np.inf  # 前者用于CART,后者用于ID3和C.45
        best_feature, best_value = -1, 0  # 定义最佳特征索引和特征值
        for feature in range(n):
            if self.algorithm == 'ID3':
                newdelta_info = self.cal_entroy_gain(base_ent, dataset, feature)
                if newdelta_info > delta_info:
                    best_feature = feature
                    delta_info = newdelta_info
            elif self.algorithm == 'C4.5':
                newdelta_info = self.cal_entroy_gain_rate(base_ent, dataset, feature)
                if newdelta_info > delta_info:
                    best_feature = feature
                    delta_info = newdelta_info
            elif self.algorithm == 'CARTcla':
                new_gini, value = self.cal_split_gini(dataset, feature)
                if new_gini < delta_gini:
                    delta_gini = new_gini
                    best_value = value
                    best_feature = feature
            else:  # CART回归树
                new_se, value = self.cal_split_se(dataset, feature)
                if new_se < delta_gini:
                    delta_gini = new_se
                    best_value = value
                    best_feature = feature
        if self.algorithm == 'ID3' or self.algorithm == 'C4.5':
            return best_feature
        else:
            return best_feature, best_value

    def training(self, dataset, feature_label=None):
        """
        训练模型,即生成决策树的函数.利用字典来作为树的数据结构.ID3和C4.5是N叉树,CART是二叉树
        :param dataset: 数据集m*(n+1),m为样本数,n为特征数
        :param feature_label: 索引值对应的含义列表,若没有给定,则用初始数据的索引值代替.
        :return: 字典形式的决策树
        """
        dataset = np.array(dataset)
        targets = dataset[:, -1]
        if np.unique(targets).shape[0] == 1:  # 即标签列表中只有一个类别，返回此类别
            return targets[0]
        if dataset.shape[1] == 1:  # 对应 没有特征值可分的情况
            return Counter(targets.tolist()).most_common(1)[0]
        if feature_label is None:  # 若没有给定对照表,则用初始数据的索引值代替.
            feature_label = [i for i in range(dataset.shape[1] - 1)]

        if self.algorithm == 'ID3' or self.algorithm == 'C4.5':
            best_feature = self.choose_best_feature(dataset)  # 选取最佳分类特征索引值
            best_feature_label = feature_label[best_feature]  # 获取其含义
            feature_label_copy = feature_label.copy()  # 避免对源数据的修改
            feature_label_copy.pop(best_feature)  # 因为这个表要传递给子树使用，所以删去表中的这个元素（不然会导致索引值混乱，从而无法对应正确的特征）
            mytree = {best_feature_label: {}}  # 创建根节点
            values = np.unique(dataset[:, best_feature])
            for value in values:  # 针对最佳分类特征的每一个属性值，创建子树
                sublabel = feature_label_copy[:]  # 更新 子 特征-含义 列表
                mytree[best_feature_label][value] = self.training(self.split_dataset(dataset, best_feature, value),
                                                                  sublabel)
        else:
            best_feature, best_value = self.choose_best_feature(dataset)
            best_feature_label = feature_label[best_feature]
            mytree = dict()
            mytree['FeatLabel'] = best_feature_label  # 记录结点选择的特征
            mytree['FeatValue'] = best_value  # 记录结点选择的特征的值
            l_set, r_set = self.split_dataset(dataset, best_feature, best_value)
            mytree['left'] = self.training(l_set, feature_label)  # 构建左子树
            mytree['right'] = self.training(r_set, feature_label)  # 构建右子树
        return mytree

    def predict(self, tree, test_data, feature_label=None):
        """
        使用训练好的决策树,对单个待测样本进行预测.如果要预测一个数据集,可以把数据集拆开来一个一个的进行预测再组合起来.
        :param tree: 训练好的决策树
        :param test_data: 待测样本1*n
        :param feature_label: 索引值对应的含义列表,若没有给定,则用初始数据的索引值代替.
        :return: 预测结果
        """
        if not isinstance(tree, dict):  # 终止条件,意味着到达叶子结点,返回叶子结点的值
            return tree
        if feature_label is None:
            feature_label = [i for i in range(test_data.shape[1] - 1)]
        if self.algorithm == 'ID3' or self.algorithm == 'C4.5':
            best_feature_label = list(tree.keys())[0]  # 获取特征-含义对照表的值
            best_feature = feature_label.index(best_feature_label)  # 获取特征的索引值
            sub_tree = tree[best_feature_label]  # 获取子树
            temp_sort=[]
            temp_min=[]
            #####以特征值相近的数值代替，精度为0.0001
            if test_data[0][best_feature] not in sub_tree:
                for key,value in sub_tree.items():
                    temp=abs(key-test_data[0][best_feature])
                    temp_min.append(temp)
                    #if temp < 0.1:
                    #if ((key-test_data[0][best_feature])<0.001):
                        #temp_sort.append(value)
                #print(Counter(temp_sort).most_common(1)[0][0])
                a=temp_min.index(min(temp_min))
                index=0
                num=0
                for key, value in sub_tree.items():
                    index=index+1
                    if index==a:
                        num=value
                value_of_feat=num
                #value_of_feat=Counter(temp_sort).most_common(1)[0][0]
            else:
                value_of_feat = sub_tree[test_data[0][best_feature]]  # 找到测试样本相应特征值对应的子树,遍历该子树         #修改后的：加了[0]！！！
            return self.predict(value_of_feat, test_data, feature_label)
        else:
            best_feature_label = tree['FeatLabel']
            best_feature = feature_label.index(best_feature_label)
            if self.algorithm == 'CARTcla':  # CART分类树
                if test_data[0][best_feature] == tree['FeatValue']:
                    return self.predict(tree['left'], test_data, feature_label)
                else:
                    return self.predict(tree['right'], test_data, feature_label)
            else:  # CART回归树
                if test_data[0][best_feature] <= tree['FeatValue']:
                    return self.predict(tree['left'], test_data, feature_label)
                else:
                    return self.predict(tree['right'], test_data, feature_label)

    def prune(self, tree, test_data):
        """
        利用测试集,对生成树进行后剪枝(CART回归树)
        :param tree: 训练好的决策树
        :param test_data: 测试集数据m*(n+1),带标签列
        :return: 剪枝后的决策树
        """

        def istree(tr):  # 判断是否为决策树
            return isinstance(tr, dict)

        def getmean(tr):  # 返回决策树所有叶子结点的均值
            if istree(tr['left']):
                tr['left'] = getmean(tr['left'])
            if istree(tr['right']):
                tr['right'] = getmean(tr['right'])
            return (tr['left'] + tr['right']) / 2

        left = right = None
        if self.algorithm == 'CARTreg':
            if not test_data:  # 如果测试集为空,则对决策树做塌陷处理,返回树的叶子结点的均值
                return getmean(tree)
            if istree(tree['left']) or istree(tree['right']):
                left, right = self.split_dataset(test_data, tree['FeatLabel'], tree['FeatValue'])
            if istree(tree['left']):
                tree['left'] = self.prune(tree['left'], left)  # 遍历左子树
            if istree(tree['right']):
                tree['right'] = self.prune(tree['right'], right)  # 遍历右子树
            if not istree(tree['left']) and not istree(tree['right']):  # 抵达叶子结点
                left, right = self.split_dataset(test_data, tree['FeatLabel'], tree['FeatValue'])
                error_nomerge = np.sum(np.power(left[:, -1] - tree['left'], 2)) + \
                                np.sum(np.power(right[:, -1] - tree['right'], 2))
                tree_mean = (tree['left'] + tree['right']) / 2
                error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
                if error_merge <= error_nomerge:  # 比较合并后与合并前,测试数据的误差,那个更小
                    return tree_mean
                else:
                    return tree
            return tree

def load_excel(path):
    result_array=[]
    # 读取初始数据
    data = xlrd.open_workbook(path)
    table = data.sheet_by_index(0)
    # table.nrows表示总行数
    for i in range(table.nrows):
        # 读取每行数据，保存在line里面，line是list
        line = table.row_values(i)
        # 将line加入到result_array中，result_array是二维list
        result_array.append(line)
    # 将result_array从二维list变成数组
    result_array = np.array(result_array)
    return result_array

def test():
    data_normal = load_excel('E:/exp/weld detect/normal.xlsx')
    data_sunken = load_excel('E:/exp/weld detect/sunken.xlsx')
    data_undercut = load_excel('E:/exp/weld detect/undercut.xlsx')
    data_pore = load_excel('E:/exp/weld detect/pore.xlsx')
    data_burnthrough = load_excel('E:/exp/weld detect/burnthrough.xlsx')
    y1 = np.zeros((data_normal.shape[0], 1))
    c1 = np.concatenate((data_normal, y1), axis=1)  # 正常焊缝

    y2 = np.ones((data_sunken.shape[0], 1)) * 1
    c2 = np.concatenate((data_sunken, y2), axis=1)  # 凹陷焊缝

    y3 = np.ones((data_undercut.shape[0], 1)) * 2
    c3 = np.concatenate((data_undercut, y3), axis=1)  # 咬边焊缝

    y4 = np.ones((data_pore.shape[0], 1)) * 3
    c4 = np.concatenate((data_pore, y4), axis=1)  # 气孔焊缝

    y5 = np.ones((data_burnthrough.shape[0], 1)) * 4
    c5 = np.concatenate((data_burnthrough, y5), axis=1)  # 焊穿焊缝

    all_data = np.concatenate((c1, c2, c3, c4, c5), axis=0)
    np.random.shuffle(all_data)
    """使用sklearn的鸢尾花数据集和生成的回归数据集分别对分类模型和回归模型测试"""
    #dataset1 = datasets.load_iris()
    #dataset1 = np.concatenate((dataset1['data'], dataset1['target'].reshape(-1, 1)), axis=1)
    dataset1=all_data
    test_data=all_data[812:]
    #dataset2 = datasets.make_regression()
    #dataset2 = np.concatenate((dataset2[0], dataset2[1].reshape(-1, 1)), axis=1)
    dt1 = DecisionTree(algorithm='ID3')
    dt2 = DecisionTree(algorithm='C4.5')
    dt3 = DecisionTree(algorithm='CARTcla')
    dt4 = DecisionTree(algorithm='CARTreg')
    #print(dt1.training(dataset1))
    #print(dt2.training(dataset1))
    #print(dt3.training(dataset1))
    #print(dt4.training(dataset2))
    ooo = dt1.training(dataset1[:812])
    ppp=ooo
    num=0

    for i in range(len(test_data)):
        if ((test_data[i][-1]==dt1.predict(ooo,test_data[i:i+1]))):
            num=num+1
        else:
            pass
    print(num/len(test_data))
    return dataset1,all_data,ppp


a,b,oo=test()
s2=datetime.datetime.now()
print("用时：" + str((s2 - s1).seconds) + 's' + str(round((s2 - s1).microseconds / 1000)) + 'ms')
print((s2-s1).microseconds)
import numpy as np
import xlrd
import os
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.tree import export_graphviz
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import datetime

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals)/ranges
    return normData

# 0.读入数据
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

if __name__ == "__main__":
    # 开始训练
    start_time = datetime.datetime.now()
    os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin'
    # 读取数据
    data_normal = load_excel('E:/exp/weld detect/normal.xlsx')
    data_sunken = load_excel('E:/exp/weld detect/sunken.xlsx')
    data_undercut = load_excel('E:/exp/weld detect/undercut.xlsx')
    data_pore = load_excel('E:/exp/weld detect/pore.xlsx')
    data_burnthrough = load_excel('E:/exp/weld detect/burnthrough.xlsx')
    #test_weld = load_excel('E:/exp/weld detect/result_test.xlsx')
    # 生成数据集
    y11 = np.array([1])
    y1 = np.tile(y11,(data_normal.shape[0],1))
    #y11 = np.zeros((data_normal.shape[0], 1))
    c1 = np.concatenate((data_normal, y1), axis=1)  # 正常焊缝

    y22 = np.array([2])
    y2 = np.tile(y22,(data_sunken.shape[0],1))
    #y2 = np.ones((data_sunken.shape[0], 1)) * 1
    c2 = np.concatenate((data_sunken, y2), axis=1)  # 凹陷焊缝

    y33 = np.array([3])
    y3 = np.tile(y33,(data_undercut.shape[0],1))
    #y3 = np.ones((data_undercut.shape[0], 1)) * 2
    c3 = np.concatenate((data_undercut, y3), axis=1)  # 咬边焊缝

    y44 = np.array([4])
    y4 = np.tile(y44,(data_pore.shape[0],1))
    #y4 = np.ones((data_pore.shape[0], 1)) * 3
    c4 = np.concatenate((data_pore, y4), axis=1)  # 气孔焊缝

    y55 = np.array([5])
    y5 = np.tile(y55,(data_burnthrough.shape[0],1))
    #y5 = np.ones((data_burnthrough.shape[0], 1)) * 4
    c5 = np.concatenate((data_burnthrough, y5), axis=1)  # 焊穿焊缝
#######load samples##
    # np.random.shuffle(c1)
    # np.random.shuffle(c2)
    # np.random.shuffle(c3)
    # np.random.shuffle(c4)
    # np.random.shuffle(c5)
    # yuan = np.concatenate((c1[:75,:], c2[:75,:], c3[:75,:], c4[:75,:], c5[:75,:]), axis=0)
#########
    #all_data=load_excel('E:/temp.xlsx')
    all_data = np.concatenate((c1, c2, c3, c4, c5), axis=0)
    np.random.shuffle(all_data)
    # train_data_x1 = all_data[:1089, :8].T
    # train_data_y1 = all_data[:1089, 8:13].T
    # train_data_x2 = all_data[1218:, :8].T
    # train_data_y2 = all_data[1218:, 8:13].T
    # train_data_x=np.concatenate((train_data_x1,train_data_x2),axis=1)
    # train_data_y = np.concatenate((train_data_y1, train_data_y2), axis=1)
    train_data_x = all_data[:812, :8].T
    train_data_y = all_data[:812, 8:13].T  # 训练集
    # #test_data_x=test_weld[:,:8].T
    # #test_data_y=test_weld[:,8:13].T
    test_data_x = all_data[812:, :8].T
    test_data_y = all_data[812:, 8:13].T  # 测试集
    # test_data_x = all_data[1089:1218, :8].T
    # test_data_y = all_data[1089:1218, 8:13].T
    train_data_y = train_data_y.astype('uint8')
    train_data_x=noramlization(train_data_x)
    test_data_x=noramlization(test_data_x)


    traffic_feature = all_data[:,:8]
    traffic_target = all_data[:, 8:9]
    # scaler = StandardScaler()  # 标准化转换
    # scaler.fit(traffic_feature)  # 训练标准化对象
    # traffic_feature = scaler.transform(traffic_feature)  # 转换数据集
    # feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target,
    #                                                                           test_size=0.3, random_state=0)
    # # clf = RandomForestClassifier(criterion='entropy')
    # clf = RandomForestClassifier()
    # clf.fit(feature_train, target_train)
    # predict_results = clf.predict(feature_test)
    # print(accuracy_score(predict_results, target_test))
    # conf_mat = confusion_matrix(target_test, predict_results)
    # print(conf_mat)
    # print(classification_report(target_test, predict_results))
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # zzz = iris.feature_names
    X= traffic_feature
    y=traffic_target

    scaler = StandardScaler()  # 标准化转换
    scaler.fit(X)  # 训练标准化对象
    X = scaler.transform(X)  # 转换数据集
    feature_train, feature_test, target_train, target_test = train_test_split(X, y,
                                                                              test_size=0.3, random_state=0)
    # clf = RandomForestClassifier(criterion='entropy')
    clf = DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=2)
    clf2 = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=3)
    rlf = RandomForestClassifier(n_estimators=1,oob_score=True, max_depth=4)
    rlf.fit(feature_train, target_train)
    predict_results = rlf.predict(feature_test)
    print(accuracy_score(predict_results, target_test))
    end_time = datetime.datetime.now()
    print("用时：" + str((end_time - start_time).seconds) + 's' + str(round((end_time - start_time).microseconds / 1000)) + 'ms')
    conf_mat = confusion_matrix(target_test, predict_results)
    print(conf_mat)
    print(classification_report(target_test, predict_results))

#   result
#     score_c = clf.score(feature_test, target_test)
#     score_r = rlf.score(feature_test, target_test)
#     print("Single Tree:{}".format(score_c) , "Random Forest:{}".format(score_r) )

    rfc_s = cross_val_score(rlf, feature_test, target_test, cv=10)
    clf_s = cross_val_score(clf, feature_test, target_test, cv=10)
    clf2_s = cross_val_score(clf2, feature_test, target_test, cv=10)

    plt.plot(range(1, 11), rfc_s, label="RandomForest")
    plt.plot(range(1, 11), clf_s, label="Decision Tree ID3")
    plt.plot(range(1, 11), clf2_s, label="Decision Tree CART")
    plt.ylim((0, 1))
    plt.legend()
    plt.show()


    feature_importances = rlf.feature_importances_
    f, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(rlf.feature_importances_)), rlf.feature_importances_)
    ax.set_title("Feature Importances")
    f.show()
    result_=rlf.oob_score_
    # 树状图

    for idx, estimator in enumerate(rlf.estimators_):
        # 导出dot文件
        export_graphviz(estimator,
                        out_file='tree{}.dot'.format(idx),
                        feature_names=['Number of profile points','Defects width','Defects height',
                                       'Defects slope','Peak valley index','Defect index','Kurtosis cofficient','Standard deviation'],
                        class_names=['normal seam','dent','undercut','pore','burn through'],
                        rounded=True,
                        proportion=False,
                        precision=2,
                        filled=True)
        # 转换为png文件
        os.system('dot -Tpng tree{}.dot -o tree{}.png'.format(idx, idx))

    # models = RandomForestClassifier(n_estimators=1,oob_score=True, max_depth=5)
    # models.fit(X, y)
    # feature_importances = models.feature_importances_
    # f, ax = plt.subplots(figsize=(7, 5))
    # ax.bar(range(len(models.feature_importances_)), models.feature_importances_)
    # ax.set_title("Feature Importances")
    # f.show()
    # result_=models.oob_score_
    # # 循环打印每棵树
    # for idx, estimator in enumerate(models.estimators_):
    #     # 导出dot文件
    #     export_graphviz(estimator,
    #                     out_file='tree{}.dot'.format(idx),
    #                     feature_names=['Number of profile points','Defects width','Defects height',
    #                                    'Defects slope','Peak valley index','Defect index','Kurtosis cofficient','Standard deviation'],
    #                     class_names=['normal','dent','undercut','porosity','burn through'],
    #                     rounded=True,
    #                     proportion=False,
    #                     precision=2,
    #                     filled=True)
    #     # 转换为png文件
    #     os.system('dot -Tpng mytree{}.dot -o mytree{}.png'.format(idx, idx))

#   评估器数量
#     scorel = []
#     for i in range(0, 60):
#         rfc = RandomForestClassifier(n_estimators=i + 1,max_depth=3)
#         score = cross_val_score(rfc, feature_test, target_test, cv=10).mean()
#         scorel.append(score)
#     #print(max(scorel), (scorel.index(max(scorel)) * 10) + 1)
#     plt.figure(figsize=[20, 5])
#     plt.plot(range(0, 60), scorel)
#     plt.show()


# #   树的深度
#     scorel_tree = []
#     for i in range(0, 20, 1):
#         rfc = RandomForestClassifier(n_estimators=1,max_depth=i+1)
#         score = cross_val_score(rfc, X, y, cv=10).mean()
#         scorel_tree.append(score)
#     # print(max(scorel), (scorel.index(max(scorel)) * 10) + 1)
#     plt.figure(figsize=[20, 5])
#     my_x_ticks = np.arange(0, 20, 2)
#     my_y_ticks = np.arange(0, 1.1, 0.1)
#     plt.xticks(my_x_ticks)
#     plt.yticks(my_y_ticks)
#     plt.plot(range(0, 20, 1), scorel_tree)
#     plt.show()



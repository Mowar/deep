import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deeprs.utils import LossHistory
from deeprs.models import DeepFM

if __name__ == "__main__":

    import os
    print(os.getcwd())
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    data = pd.read_csv('./datasets/criteo_sample.txt')
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    # feature_name = ["label"]
    # feature_name.extend(dense_features)
    # feature_name.extend(sparse_features)
    # data = pd.read_csv("./datasets/dac_sample.txt",sep="\t",header=None, names=feature_name)

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)
    data[dense_features] = data[dense_features].astype({feats:"float64" for feats in dense_features})
    target = ['label']

    # 1.类别特征label编码 和 数值特征0-1缩放
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.统计每个类别特征的取值数和记录数值特征的字段名
    sparse_feature_dict = {feat: data[feat].nunique()
                           for feat in sparse_features}
    dense_feature_list = dense_features

    # 3.生成模型的输入数据集
    model_input = [data[feat].values for feat in sparse_feature_dict] + \
        [data[feat].values for feat in dense_feature_list]  # + [data[target[0]].values]

    # 4.定义模型 编译和训练
    model = DeepFM({"sparse": sparse_feature_dict,
                    "dense": dense_feature_list}, final_activation='sigmoid')

    history = LossHistory("binary_crossentropy")

    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    model_train = model.fit(model_input, data[target].values,

                        batch_size=256, epochs=500, verbose=2, validation_split=0.2, callbacks=[history])
    history.loss_plot()
    print("demo done")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deeprs import DeepFM

if __name__ == "__main__":

    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    data = pd.read_csv("./datasets/movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]
    target = ['rating']

    # 1.类别特征label编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.统计每个类别特征的取值数
    sparse_feature_dim = {feat: data[feat].nunique()
                          for feat in sparse_features}

    # 3.生成模型的输入数据集
    model_input = [data[feat].values for feat in sparse_feature_dim]

    # 4.定义模型 编译和训练
    model = DeepFM({"sparse": sparse_feature_dim, "dense": []},
                   final_activation='linear')

    model.compile("adam", "mse", metrics=['mse'],)
    history = model.fit(model_input, data[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2,)

    print("demo done")

from sklearn.preprocessing import StandardScaler
from experiment_utils import *
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter

# Experiment Setting
settings = {
    "random_seed": 22,
    "k_fold_split": 10,
    "train_test_ratio": 0.2,
    "repeats": 10,
    "is_run_sampling": False,
}

# Random
np.random.seed(settings["random_seed"])

metrics = {
    "knn": init_metric_temp(),
    "svm": init_metric_temp(),
    "xgboost": init_metric_temp(),
    "random_forest": init_metric_temp(),
    "lightgbm": init_metric_temp(),
}
params = {
    "svm": generate_params({'kernel': ['linear', 'poly'], 'degree': [2, 3, 4, 5]}),
    "knn": generate_params({"n_neighbors": [5, 10, 15], "leaf_size": [10, 50, 100]}),
    "random_forest": generate_params({"n_estimators": [50, 100, 200], "min_samples_leaf": [5, 10, 50], "random_state": [50]}),
    "xgboost": generate_params({"learning_rate": [0.1, 0.2, 0.5], "n_estimators": [50, 100, 500], "random_state": [50]}),
    "lightgbm": generate_params({"learning_rate": [0.1, 0.2, 0.5], "n_estimators": [50, 100, 500], "random_state": [50]}),
}


def random_sampling(X, y):
    print("Class dis before sampling:", sorted(Counter(y).items()))
    undersample = SMOTE()
    # Data preprocessing
    X_sampling, y_sampling = undersample.fit_resample(X, y)
    print("Class dis after sampling:", sorted(Counter(y_sampling).items()))
    return shuffle(pd.concat([X_sampling, y_sampling], axis=1))


def run(X_train, X_test, y_train, y_test, metrics, model, params=None, isSTD=True):
    if isSTD:
        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)  # standardizing the data
        X_test = standard_scaler.transform(X_test)
    a_tst, p_tst, r_tst, f1_tst, trained_model, param = model_running(X_train,
                                                                      X_test,
                                                                      y_train,
                                                                      y_test,
                                                                      settings["k_fold_split"],
                                                                      model,
                                                                      params)
    metric_recording(metrics, a_tst, p_tst, r_tst, f1_tst, param)
    return trained_model


def load_data(seed, path, feature_list=None):
    all_data = pd.read_csv(path)
    all_data = shuffle(all_data, random_state=seed)
    popular_label = all_data.groupby('label').count().reset_index()
    popular_label = popular_label[popular_label['address'] > 50]
    popular_label = set(popular_label["label"].values)
    all_data = all_data[all_data["label"].isin(popular_label)]
    X = all_data.drop(["address", "label"], axis=1)
    if feature_list:
        X = X[feature_list]
    y = all_data['label']
    return X, y


def lists_to_dict(keys, values):
    return {keys[i]: values[i] for i in range(len(keys))}


def opcodes_based_experiment():
    settings["is_run_sampling"] = False
    model_experimenting("OPCODE", "opcode_feature_dataset.csv")


def exchange_based_experiment():
    settings["is_run_sampling"] = False
    model_experimenting("EXCHANGE", "exchange_feature_dataset.csv")


def model_experimenting(prefix, path, features_list=None):
    experiment_id, curves_path, data_splits_path, metrics_path, models_path = setup(prefix, "experiment_results", settings)
    X, y = load_data(seed=100, path=path, feature_list=features_list)
    is_header = True
    for run_id in range(settings["repeats"]):
        if settings["is_label_mapping"]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings["train_test_ratio"], stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings["train_test_ratio"])
        # train_test_split_saving(data_splits_path, run_id, X_train, X_test, y_train, y_test)

        columns = X_train.columns

        # sampling
        if settings["is_run_sampling"]:
            data_sampling = random_sampling(X_train, y_train)
            X_train = data_sampling.drop(["label"], axis=1)
            y_train = data_sampling['label']

        # SVM
        # svm = SVC(kernel='poly', C=1, degree=3)
        run(X_train, X_test, y_train, y_test, metrics["svm"], SVC, params["svm"])
        # run(X_train, X_test, y_train, y_test, metrics["svm"], SVC)

        # KNN
        # knn = KNeighborsClassifier()
        run(X_train, X_test, y_train, y_test, metrics["knn"], KNeighborsClassifier, params["knn"])
        # run(X_train, X_test, y_train, y_test, metrics["knn"], KNeighborsClassifier)

        # Random Forest
        # random_forest = RandomForestClassifier(random_state=seed)
        run(X_train, X_test, y_train, y_test, metrics["random_forest"], RandomForestClassifier, params["random_forest"])
        # run(X_train, X_test, y_train, y_test, metrics["random_forest"], RandomForestClassifier)

        # XGBoost
        # xgboost = xgb.XGBClassifier(use_label_encoder=False, random_state=seed)
        xgboost = run(X_train, X_test, y_train, y_test, metrics["xgboost"], xgb.XGBClassifier, params["xgboost"])
        # xgboost = run(X_train, X_test, y_train, y_test, metrics["xgboost"], xgb.XGBClassifier)
        xgboost_important_features = pd.DataFrame([lists_to_dict(columns, xgboost.feature_importances_)])
        xgboost_important_features.to_csv(os.path.join(metrics_path, "xgboost_important_features.csv"),
                                          mode='a',
                                          header=is_header,
                                          index=False)

        # LightGBM
        # lightgbm = LGBMClassifier(random_state=seed)
        lightgbm = run(X_train, X_test, y_train, y_test, metrics["lightgbm"], LGBMClassifier, params["lightgbm"])
        # lightgbm = run(X_train, X_test, y_train, y_test, metrics["lightgbm"], LGBMClassifier)
        lightgbm_important_features = pd.DataFrame([lists_to_dict(columns, lightgbm.feature_importances_)])
        lightgbm_important_features.to_csv(os.path.join(metrics_path, "lightgbm_important_features.csv"),
                                           mode='a',
                                           header=is_header,
                                           index=False)

        is_header = False
    print("Exporting metrics records")
    save_metrics(metrics_path, metrics)


if __name__ == "__main__":
    print("EXCHANGE BASED BINARY CLASSIFICATION")
    exchange_based_experiment()
    print("OPCODE BASED BINARY CLASSIFICATION")
    opcodes_based_experiment()

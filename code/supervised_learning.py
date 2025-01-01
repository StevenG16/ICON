import data_preprocessing
import random
from sklearn import tree, linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import pickle


hyperparams = {
    "DecisionTree":{
        "criterion":["gini", "entropy", "log_loss"],
        "splitter":["best"],
        "max_depth":[None, 1, 2, 5, 10, 25, 50],
        "min_samples_split": [2, 5, 10, 25, 50, 100],
        "min_samples_leaf": [1, 2, 5, 10, 25, 50, 100], 
    },
    "LogisticRegression": {
        "C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "solver":["lbfgs", "newton-cg", "sag", "saga"],
        "max_iter": list(range(100,1000,100))
    },
    "RandomForest": {
        "n_estimators": [5, 10, 25, 50, 100],
        "criterion":["gini", "entropy", "log_loss"],
        "max_depth":[None, 1, 2, 5, 10, 25, 50],
        "min_samples_split": [2, 5, 10, 25, 50, 100],
        "min_samples_leaf": [1, 2, 5, 10, 25, 50, 100], 
    },
    "GradientBoosting": {
        "loss":["log_loss", "exponential"],
        "learning_rate":[0.0, 0.1, 1, 10],
        "n_estimators":[10, 50, 100, 250],
        "max_depth":[None, 1, 2, 5, 10, 25],
        "criterion":["friedman_mse", "squared_error"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
    }
}


best_hyperparameters = {
    "DecisionTree": {'criterion': 'log_loss', 'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 2, 'splitter': 'best'},
    "LogisticRegression": {'C': 1000, 'max_iter': 900, 'solver': 'sag'},
    "RandomForest": {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100},
    "GradientBoosting": {'criterion': 'squared_error', 'learning_rate': 1, 'loss': 'log_loss', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 250}
}



class Dataset(object):
    
    def __init__(self, data, test_prob = 0.20, target_index = 0, seed = None):

        if seed:
            random.seed(seed)


        train_targets = []
        test_targets = []
        train = []
        test = []
        random.shuffle(data)
        for example in data:
            target = example.pop(target_index)
            is_test = random.random() < test_prob
            if is_test:
                test.append(example)
                test_targets.append(target)
            else:
                train.append(example)
                train_targets.append(target)
        self.train = train
        self.test = test
        self.train_targets = train_targets
        self.test_targets = test_targets



def hyperparameter_tuning(dataset, model, model_name):
    search = GridSearchCV(model, hyperparams[model_name], cv=5, n_jobs=-1, verbose=3)
    search.fit(dataset.train, dataset.train_targets)
    return search.best_params_
    

def model_validation(dataset, model, model_name):
    results = {}
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    kfold = KFold(n_splits = 10)
    for metric in metrics:
        pass

    return


def save_item(filepath, item):
    with open(filepath, "wb") as f:
        pickle.dump(item, f)


def load_item(filepath):
    tree = None
    with open(filepath, "rb") as f:
        tree = pickle.load(f)
    return tree


def load_and_test():
    tree_fp = "tree.pkl"
    clf = load_item(tree_fp)
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, fontsize=4, max_depth=4)
    plt.savefig("imgs/tree.png", dpi = 500)


def tune_models():
    print("[*] Querying the dataset...")
    data = data_preprocessing.query_to_dataset(data_preprocessing.data_files["small_prolog_clustered"], data_preprocessing.queries["factors_all_clustered"])
    print("[+] Dataset successfully queried")
    dataset = Dataset(data, target_index=-1)
    print("Targets: ", [dataset.train_targets[i] for i in range(50) ], " ...")

    models = [(tree.DecisionTreeClassifier(), "DecisionTree"), (linear_model.LogisticRegression(), "LogisticRegression"),
                (RandomForestClassifier(), "RandomForest"), (GradientBoostingClassifier(), "GradientBoosting")]

    optimal_parameters = {}
    for model, model_name in models:
        print(f"[*] Searching the best hyperparameters configuration for {model_name}...")
        best_results = hyperparameter_tuning(dataset, model, model_name)
        print(f"[+] Optimal hyperparameters configuration found for {model_name}:\n{best_results}")
        optimal_parameters[model_name] = best_results
    save_item("params_config.pkl", optimal_parameters)


def main():
    optimal_parameters = load_item("params_config.pkl")
    print(optimal_parameters)

if __name__ == "__main__":
    main()




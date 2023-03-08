# https://towardsdatascience.com/looking-beyond-feature-importance-37d2807aaaa7
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from graphviz import Source
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib as mpl


def feature_importance(model, X):
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.keys())
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.show()


# def decision_tree(model, X):
#     timestamp = time.monotonic_ns()
#     path = 'tree_' + str(timestamp) + '.dot'
#     export_graphviz(model, out_file=path, feature_names=X.columns,
#                     impurity=False, filled=True)
#     s = Source.from_file(path)
#     s.render('tree', format='jpg', view=True)
#     # https://scikit-learn.org/stable/modules/tree.html


def chart(results_arr):
    plt.plot(list(results_arr.keys()), list(results_arr.values()))
    plt.show()


# more: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
# https://community.ibm.com/community/user/cloudpakfordata/blogs/harris-yang1/2021/05/26/scikit-learn-churn-model-cpd35
# https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv
def simple_roc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC', fontsize=18)


def simple_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()

    return cm


def simple_heatmap(df, figsize=(20, 18), fontsize=12):
    data_correlations = df.corr()
    plt.subplots(figsize=figsize)
    sns.heatmap(data_correlations, cmap='Blues', annot=True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()


def simple_correlations(df, target_name):
    cols = df.columns
    for label in cols:
        if label != target_name:
            sns.catplot(x=label, y=target_name, data=df)


def simple_features_overview(df):
    for label in df.columns:
        sns.catplot(x=label, data=df, height=3, aspect=4)


def plot_bars(name, data_x, data_y, labelx, labely, groupbyx=True):
    import matplotlib.pyplot as plt
    df = pd.DataFrame({"data_x": data_x, "data_y": data_y})
    if groupbyx:
        df.groupby(['data_x']).sum()
    # print(df.head())
    # plt.rcParams.update({'figure.autolayout': True})
    plt.plot(df)
    plt.title(name, fontsize=25)
    plt.xlabel(labelx, fontsize=15)
    plt.ylabel(labely, fontsize=15)
    plt.legend()
    plt.show()


def plot_predictions(y_valid, y_pred):
    scale = range(len(y_valid))

    plt.plot(scale, y_valid, label="original")
    plt.plot(scale, y_pred, label="predicted")
    plt.title("Prediction vs validation")
    plt.legend()
    fig = plt.figure(figsize=(40, 15))
    # plt.scatter(y_valid, y_pred)
    plt.plot()


def plot_history(history):
    history_df = pd.DataFrame(history.history)
    fig, ax = plt.subplots()
    ax.plot(history_df)


def plot_model_history(x1, x2, label1, label2):
    mpl.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_title("Baseline and predictions", color='C0')
    ax.plot(range(len(x1)), x1, color="#f00", label=label1)
    ax.plot(range(len(x2)), x2, color="#00f", label=label2)
    plt.legend()


def plot_pandas_dataframe(df, label):
    mpl.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 7))
    # ax.set_title("Baseline and predictions", color='C0')
    ax.plot(range(len(df)), df, color="#00a", label=label)
    # ax.plot(range(len(x2)), x2, color="#00f", label=label2)
    plt.legend()


def plot_y(y):
    scale = range(len(y))

    plt.plot(scale, y)
    plt.legend()
    fig = plt.figure(figsize=(40, 15))

from sklearn import datasets
from sklearn import svm
import pickle
import os


def train():
    """ This code is from: https://scikit-learn.org/stable/tutorial/basic/tutorial.html """
    iris = datasets.load_iris()

    svm_model = svm.SVC(gamma=0.001, C=100.0)

    svm_model.fit(iris.data[:-1], iris.target[:-1])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = open(os.path.join(dir_path, "model_files", "svc_model.pickle"), 'wb')
    pickle.dump(svm_model, file)
    file.close()


if __name__ == "__main__":
    train()

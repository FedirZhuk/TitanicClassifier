import tarfile
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

# Wczytanie pliku
tarball_path = Path("datasets/titanic.tgz")

with tarfile.open(tarball_path) as titanic_tarball:
    titanic_tarball.extractall(path="datasets")

train_data, test_data = [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]

# print(train_data.head())
# print(test_data.head())

# Wyznaczmy jawnie kolumnę PassengerId jako kolumnę indeksu:
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

# Uzyskajmy więcej informacji na temat brakujących danych:
# print(train_data.info())
# print('Mediana wieku kobiet: ' + str(train_data[train_data["Sex"]=="female"]["Age"].median()))

# Przyjrzyjmy się atrybutom numerycznym:
# print(train_data.describe())
# print(train_data["Survived"].value_counts())

# Przyjrzyjmy się teraz wszystkim atrybutom kategorialnym:
# print(train_data["Pclass"].value_counts())
# print(train_data["Sex"].value_counts())
# print(train_data["Embarked"].value_counts())

# train_data["AgeBucket"] = train_data["Age"] // 15 * 15
# train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
#
# train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
# train_data[["RelativesOnboard", "Survived"]].groupby(
#     ['RelativesOnboard']).mean()

def add_age_bucket(X):
    return X[:,[0]] // 15 * 15

def add_relatives_onboard(X):
    return X[:,[0]] + X[:,[1]]

def age_bucket_name(function_transformer, feature_names_in):
    # Provide a name for the output feature of the age bucket transformation
    return ['age_bucket']

def relatives_onboard_name(function_transformer, feature_names_in):
    return ['relatives_onboard']

def age_pipline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(add_age_bucket, feature_names_out=age_bucket_name),
        StandardScaler())

def relatives_pipline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(add_relatives_onboard, feature_names_out=relatives_onboard_name),
        StandardScaler())

# Zbudujmy teraz nasze potoki wstępnego przetwarzania danych, począwszy od potoku zajmującego się atrybutami numerycznymi:
num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

# Teraz możemy zbudować potok dla atrybutów kategorialnych:
cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder()),
    ])

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

# Połączmy w końcu potoki numeryczny i kategorialny:
preprocess_pipeline = ColumnTransformer([
        ("age", age_pipline(), ["Age"]),
        ("rel", relatives_pipline(), ["SibSp", "Parch"]),
        # ("drop", drop_notused(), ["SibSp", "Parch", "Embarked", "Age"])
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),

    ])



# Przetwarzanie danych treningowych
X_train = preprocess_pipeline.fit_transform(train_data)
# Nie zapominajmy o etykietach:
y_train = train_data["Survived"]

# Możemy już przystąpić do uczenia klasyfikatora. Zacznijmy od klasy RandomForestClassifier:
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print('Random Forest ' + str(forest_scores.mean()))

# Wypróbujmy klasę SVC:
svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print('SVC ' + str(svm_scores.mean()))

plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Las losowy"))
plt.ylabel("Dokładność")
plt.show()

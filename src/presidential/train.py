import spacy
from presidential.preprocess import segments, labels, REP_CANDIDATES, DEM_CANDIDATES
from presidential.features import Featurizer

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


classes = set(DEM_CANDIDATES + REP_CANDIDATES)
nlp = spacy.load("en_core_web_md")
featurizer = Featurizer()


list_of_feature_dicts = []
for d in segments:
    feature_dict = featurizer.featurize(nlp(d))
    list_of_feature_dicts.append(feature_dict)


# encode labels with value between 0 and n_classes-1
le = LabelEncoder()
le.fit([c.lower() for c in classes])
y = le.transform(labels)

# vectorize
vec = DictVectorizer()
X = vec.fit_transform(list_of_feature_dicts)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# transform categorical labels to numeric ones
# y_train = le.transform(annotator_labels)
# y_test = le.transform(intern_labels)

clf = LogisticRegression(
    random_state=42, class_weight="balanced", solver="lbfgs", multi_class="auto"
)
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # smaller C stronger regularization
    "max_iter": [100, 1000, 10000],
}
grid_search = GridSearchCV(clf, param_grid)
grid_search.fit(X_train, y_train)
print(f"best params for log reg: {grid_search.best_params_}")
clf.set_params(**grid_search.best_params_)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(
    le.inverse_transform(y_test), le.inverse_transform(y_pred), digits=3
)

print(accuracy)
print(cr)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

xs = [[0, 0, 0.5], [1, 0.5, 1], [0.5, 2, 2]]
ys = [0, 1, 2]

clf = DecisionTreeClassifier(random_state=42)
clf.fit(xs, ys)

result = clf.predict([[0.5, 2, 2]])
print(result)

plt.figure()
tree.plot_tree(clf, filled=True)
plt.show()

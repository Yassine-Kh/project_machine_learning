class DecisionTree:

    def __init__(self, n_depths=10):
        self.cvp = ShuffleSplit(1000, train_size=2 / 3)
        self.n_depths = n_depths
        self.depths = np.linspace(1, 10, self.n_depths)

    def classification(self):
        tab_RMSE_tree = np.zeros(self.n_depths)
        for i in range(self.n_depths):
            reg_tree = DecisionTreeClassifier(max_depth=self.depths[i])
            tab_RMSE_tree[i] = np.median(
                np.sqrt(-cross_val_score(reg_tree, X_train, y_train, scoring='neg_log_loss', cv=self.cvp)))
        return tab_RMSE_tree

    def plot(self):
        plt.plot(self.depths, self.classification())
        plt.axis([0, 6, 0, 20])
        plt.show()

reg_tree = DecisionTree()
reg_tree.plot()
plt.show()
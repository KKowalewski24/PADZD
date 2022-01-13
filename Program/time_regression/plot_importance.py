import xgboost as xgb
import matplotlib.pyplot as plt

bst = xgb.Booster()
bst.load_model("./best_model.model")
bst.feature_names = xgb.DMatrix("test_data.buffer").feature_names

xgb.plot_importance(bst, max_num_features=30, grid=False)
plt.tight_layout()
plt.savefig("importance.png", dpi=200)

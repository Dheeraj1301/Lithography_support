import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_corr(df):
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.show()

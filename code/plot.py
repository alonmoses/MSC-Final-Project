import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.autolayout"] = True
import seaborn as sns

def plot_train_vs_test(train_losses, test_losses, title):
    plt.title(title)
    plt.plot(np.sqrt(train_losses), color="red")
    plt.plot(np.sqrt(test_losses), color="blue")
    

def plot_genes_read_count(data, bin_count=25, filter_count=0):
    if filter_count:
        bin_count, bin_val = np.histogram(data[data<filter_count], bins=bin_count)
    else:
        bin_count, bin_val = np.histogram(data, bins=bin_count)
    plt.figure(figsize=(12, 6))
    plt.title('gene reads histogram')
    plt.ylabel('count')
    plt.xlabel('reads')
    sns.barplot(x=bin_val[:-1].astype(int), y=bin_count)
    plt.yscale('log')
    plt.show()
    
def plot_true_vs_reconstructed_histogram(df_expressions_true, df_expressions_preds):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 6))
    df_expressions_true.expression.hist(ax=ax1)
    df_expressions_preds.expression.hist(ax=ax2)
    ax1.set_title('True Genes Expression Histogram')
    ax2.set_title('Prediction Genes Expression Histogram')
    plt.show()
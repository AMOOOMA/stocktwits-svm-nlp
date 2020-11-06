import matplotlib.pyplot as plt
import numpy as np

def mk_pca(results):
    
    x_pos = np.arange((len(results)))
    x_label = ["PCA"+str(i) for i in range(1,len(results+1))]
    
    print(results)
    plt.bar(x_pos, results, alpha=0.5)
    plt.xticks(x_pos, x_label)
    plt.ylabel('variance')
    
    plt.show()

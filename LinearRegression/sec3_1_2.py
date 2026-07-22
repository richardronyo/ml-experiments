import numpy as np
import matplotlib.pyplot as plt

from model import RSS, RSE, SE_B0, SE_B1

if __name__ == "__main__":
    #Creating a population dataset that resembles the line Y = 2+ 3X 
    n = 10000
    X = np.linspace(-1000, 1000, n)
    epsilon = np.random.normal(loc = 0.0, scale = 50, size = n) 
    Y = 2 + 3*X + epsilon
    Y_true = 2 + 3*X

    fig, axes = plt.subplots(2, 2, figsize = (10, 10))

    axes[0, 0].set_title("Scatter Plot of Population Data Points")
    axes[0, 0].scatter(X, Y, alpha = 0.15, c = 'steelblue', edgecolors = 'none')
    
    axes[0, 1].set_title("Y = 2 + 3*X")
    axes[0, 1].plot(X, Y_true, c = 'red')

    #Getting 10 sample datasets from my population dataset, and estimating the Linear Regression
    colours = plt.cm.tab10.colors
    beta0s = np.zeros(10)
    beta1s = np.zeros(10)

    beta0_CIs = []
    beta1_CIs = []
    
    for i in range(10):
        print("SAMPLE ", i) 
        n_sample = 1000
        sample_idx = np.random.choice(n, size = n_sample, replace = False)
        X_sample = X[sample_idx]
        Y_sample = Y[sample_idx]
    
        beta0, beta1, rss = RSS(X_sample, Y_sample)
               
        beta0s[i] = beta0
        beta1s[i] = beta1
        
        #Creating the Confidence Intervals by estimating Var(e) with the RSS
        rse = RSE(rss, n)
        se_b0 = SE_B0(rse, n_sample, X_sample)
        se_b1 = SE_B1(rse, X_sample)

        beta0_CIs.append((beta0 - se_b0, beta0 + se_b0))
        beta1_CIs.append((beta1 - se_b1, beta1 + se_b1))

        print("\tBeta0 = ", beta0)
        print("\tBeta1 = ", beta1)
        
        Y_pred = beta0 + beta1*X_sample
        axes[1, 0].set_title("Scatter Plot of Sample Data Points (n = 1000)")
        axes[1, 0].scatter(X_sample, Y_sample, alpha = 0.08, color = colours[i])
        
        axes[1, 1].set_title("Sample Linear Regressions")
        axes[1, 1].plot(X_sample, Y_pred, c = colours[i])
        

        
    plt.tight_layout()
    plt.show()
    
    #Printing the results of my sample Beta0 and Beta1 values 
    print("Sample Beta0")
    print(beta0s)
    print("95% Confidence Intervals:\n")
    print(beta0_CIs)
    print("Mean Beta0: ", np.mean(beta0s))
    print("Sample Beta1")
    print(beta1s)
    print("95% Confidence Intervals:\n")
    print(beta1_CIs)
    print("Mean Beta1: ", np.mean(beta1s))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

from scipy.optimize import minimize


mean_1 = [1,0]
mean_2 = [0,1.5]

cov_1 = [[1,0.75],[0.75,1]]
cov_2 = [[1,0.75],[0.75,1]]

def loadtraindata(mean_1, mean_2, cov_1, cov_2):
    traindata_1 = np.append(np.random.multivariate_normal(mean_1,cov_1,1500),np.zeros((1500,1)),axis=1)
    traindata_2 = np.append(np.random.multivariate_normal(mean_2,cov_2,1500),np.ones((1500,1)),axis=1)
    train_data = np.concatenate((traindata_1, traindata_2))
    return train_data

def loadtestdata(mean_1, mean_2, cov_1, cov_2):
    testdata_1 = np.append(np.random.multivariate_normal(mean_1,cov_1,500),np.zeros((500,1)),axis=1)
    testdata_2 = np.append(np.random.multivariate_normal(mean_2,cov_2,500),np.ones((500,1)),axis=1)
    test_data = np.concatenate((testdata_1, testdata_2))
    return test_data

data = loadtraindata(mean_1, mean_2, cov_1, cov_2)
data_test = loadtestdata(mean_1, mean_2, cov_1, cov_2)

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='g', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True)
    plt.show()


X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]

X_test = np.c_[np.ones((data_test.shape[0],1)), data_test[:,0:2]]
y_test = np.c_[data_test[:,2]]

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def costFunction(theta, X, y): #cross-entropy
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))       
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    grad =(1/m)*X.T.dot(h-y)
    # print(grad)
    return(grad.flatten())


def predict(theta, X, lr=0.01):
    p = sigmoid(X.dot(theta.T)) >= lr
    return(p.astype('int'))


def plotcurve(y_test,p):

    false,true,threshold = roc_curve(y_test, p)
    roc_auc = auc(false, true)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(false, true, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    x = false[1]

    plt.fill(false, true, 'c', true, false,'c',[0.0,1.0,1.0],[0.0,0.0,x],'c')
    plt.show()


initial_theta = np.zeros(X.shape[1]) #weights and bias [ 0.  0.  0.]

cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

print('Cost: \n', cost)
print('Grad: \n', grad)

res = minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':3000})

p = predict(res.x, X_test) 


x1_min, x1_max = X_test[:,1].min(), X_test[:,1].max()
x2_min, x2_max = X_test[:,2].min(), X_test[:,2].max()

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)

plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='k');

print('Accuracy {}%'.format(100*sum(p == y_test.ravel())/p.size))

plotData(data_test, 'X-axis', 'Y-axis', '1', '0')
plotcurve(y_test,p)
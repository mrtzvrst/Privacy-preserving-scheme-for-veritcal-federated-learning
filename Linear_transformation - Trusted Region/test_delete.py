from scipy.optimize import minimize, approx_fprime, NonlinearConstraint
import numpy as np
import pickle

class My_Optimization(object):
    def __init__(self, missing_features, X):
        dpas = len(missing_features)
        Xn = X[:, missing_features]
        self.K0 = np.zeros((dpas, dpas))
        for j in range(Xn.shape[0]):
            temp = np.reshape(Xn[j, :], (-1,1))
            self.K0 += np.matmul(temp, temp.T)
        self.K0/=Xn.shape[0]
        self.k = -self.K0[:,1]
        self.d = len(missing_features)
    
    def fun(self, x):
        return np.matmul(x, np.matmul(-self.K0, x))-2*np.matmul(self.k, x)
    
    def OPTIM(self):
        self.cons = lambda x:  sum(x**2)
        self.nlc = NonlinearConstraint(self.cons, 0, 1)
        self.bnds = tuple([tuple(a) for a in self.d*[[-1, 1]]])
    
        self.jac = lambda x,*args: approx_fprime(x,self.fun,epsilon=10**-10,*args)
        self.x0 = np.ones((self.d))/100
        res = minimize(self.fun, self.x0, method='trust-constr',
                        jac=self.jac,     
                        options={'gtol': 1e-10, 'disp': False, 'maxiter': 100000},
                        bounds=self.bnds,
                        constraints=self.nlc)
        return res
    
    def Matric_obtain(self):
        Matrix = np.zeros((self.d, self.d))
        for i in range(self.d):
            self.k = -self.K0[:,i]
            Matrix[i,:] = self.OPTIM().x
        return Matrix
        
        
        
        
f = open('LR_Model_No_Trans.pckl', 'rb')
model = pickle.load(f)
f.close()

X, Y = model[1], model[2]
missing_features = np.array([0,1])

M = My_Optimization(missing_features, X).Matric_obtain()
x = np.array([1/2,1/2])
res = M


np.matmul(X[:,missing_features], res)

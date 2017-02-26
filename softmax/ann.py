import numpy as np

class ANN:
    
    def __init__(self, n_in, n_hidden, n_out):
        self.W_i = np.random.rand(n_hidden, n_in)
        self.b_i = np.zeros(n_hidden)
        self.W_h = np.random.rand(n_out, n_hidden)
        self.b_h = np.zeros(n_out)

    def sigma(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def dsigma(self, x):
        s = self.sigma(x)
        return s*(1-s)
    
    def error(self, h, y):
        return np.dot(h-y, h-y)


    def fit(self, X, y):
        
        alpha = 5
        beta = 0.000
        n_iter = 10000

        activate = self.sigma
        derivate = self.dsigma
        error = self.error

        m = X.shape[0]

        prev_error = 0.0
        
        last_adjust_iter = 0
        
        for i_iter in range(n_iter):    
            
            avg_error = 0.0

            dWh_buf = np.zeros_like(self.W_h)
            dbh_buf = np.zeros_like(self.b_h)
            dWi_buf = np.zeros_like(self.W_i)
            dbi_buf = np.zeros_like(self.b_i)

            for i in range(m):
                
                x = X[i].reshape(-1).T
                z_i = np.dot(self.W_i, x) + self.b_i
                a_i = activate(z_i)

                z_h = np.dot(self.W_h, a_i) + self.b_h
                a_h = activate(z_h)
                h = a_h

                e = error(h, y[i])
                avg_error += e

                dh = -(y[i] - h) * derivate(z_h)
                di = np.dot(self.W_h.T, dh) * derivate(z_i)

                dWh = np.dot(a_h, dh.T)
                dbh = dh

                dWi = np.dot(a_i, di.T)
                dbi = di

                dWh_buf += dWh
                dbh_buf += dbh
                dWi_buf += dWi
                dbi_buf += dbi

                # self.W_h -= alpha * (dWh + beta*self.W_h)
                # self.b_h -= alpha * dbh

                # self.W_i -= alpha * (dWi + beta*self.W_i)
                # self.b_i -= alpha * dbi
            
            self.W_h -= alpha * (dWh_buf / m + beta * self.W_h)
            self.b_h -= alpha * (dbh_buf / m)
            self.W_i -= alpha * (dWi_buf / m + beta * self.W_i)
            self.b_i -= alpha * (dbi_buf / m)

            if (i_iter+1) % 100 == 0:
                print 'ITERATION %d: average error: %f'%(i_iter+1, 1.0 * avg_error / m)
                print 'alpha %f, beta %f'%(alpha, beta)

            if i_iter - last_adjust_iter > 20:
                if abs(prev_error - avg_error) / avg_error < 1e-5 or prev_error < avg_error:
                    alpha *= 0.3
                    beta *= 0.1
                    last_adjust_iter = i_iter
            
            prev_error = avg_error


    def predict(self, X):
        x = X[i].reshape(-1).T
        z_i = np.dot(self.W_i, x) + self.b_i
        a_i = activate(z_i)

        z_h = np.dot(self.W_h, a_i) + self.b_h
        a_h = activate(z_h)
        h = a_h

        return h




ann = ANN(2, 2, 1)

X = np.random.rand(1000, 2)
X[:, 0] = X[:, 0] > 0.5
X[:, 1] = X[:, 1] > 0.5
y = np.logical_and(X[:, 0], X[:, 1])


print y.shape
ann.fit(X, y)
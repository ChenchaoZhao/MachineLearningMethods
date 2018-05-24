import numpy as np
import scipy as sp

palettes = [
    '#D870AD', '#B377D9', '#7277D5', 
    '#4B8CDC', '#3BB1D9', '#3BBEB0', 
    '#3BB85D', '#82C250', '#B0C151', 
    '#F5BA42', '#F59B43', '#E7663F', 
    '#D94C42', '#655D56', '#A2A2A2']
cc_ = np.array(palettes)

class HiddenMarkovModel:
    def __init__(self, params, max_iter=500, eps=1e-6):
        
        self.max_iter = max_iter
        self.eps = eps
        
        self._p = np.array(params["initial"])
        assert self._p.sum() == 1
        self.K = self._p.size # n hidden states
        self._a = np.array(params["transition"])
        assert np.all(self._a.sum(axis=1).round(6) == np.ones(self.K))
        self._e_params = params["emission"]
        self.memo = {}
        self.posterior1 = self.posterior2 = None
        
        self._is_fitted = False
        
        
    def fit(self, observations):
        self._is_fitted = True
        self.x_ = np.array(observations)
        self.t_max = self.x_.size
        
        self._Q_ = []
        
        
        # EM
        _ = self._e_step()
        for itr in range(self.max_iter):
            self._m_step()
            is_convergent = self._e_step()
            
            if is_convergent: break
        
        if not is_convergent and itr == self.max_iter-1:
            print("EM did not converge")
        else:
            print("EM converged within {} steps".format(itr+1))
    
    def get_forcast_probability(self, T, n_sig=3, n_pts=200):
        assert T > 0 and self._is_fitted
        p_T = self.posterior1[-1]
        for t in range(T):
            p_T = p_T @ self._a
        
        if self._e_params["type"] == "normal":
            temp = []
            for k in range(self.K):
                m = self._e_params["mean"][k]
                s = self._e_params["variance"][k]
                temp += [m-n_sig*np.sqrt(s), m+n_sig*np.sqrt(s)]
            x_ = np.linspace(np.min(temp), np.max(temp), n_pts)
            predict_ = 0.0
            for k in range(self.K):
                predict_ += p_T[k] * self.__emission(x_, k)
            predict_ = np.array(predict_)
            return x_, predict_/predict_.sum()
        else:
            raise ValueError("distribution not tabulated")
            
    def predict_state(self, T, n_sig=3, n_pts=200):
        assert T > 0 and self._is_fitted
        p_T = self.posterior1[-1]
        for t in range(T):
            p_T = p_T @ self._a
        return np.argmax(p_T)
    
    def predict_observation(self, T, method="mean"):
        x_, p_ = self.get_forcast_probability(T)
        if method == "mode":
            return x_[np.argmax(p_)]
        elif method == "mean":
            return np.sum(x_*p_)
        else:
            raise ValueError("Unknown method")
    def export_params(self):
        return {"initial": self._p, "transition": self._a, "emission": self._e_params}
    
    def _e_step(self):
        # get posteriors
        self._e = np.zeros((self.t_max, self.K))
        _ = self._alpha_(self.t_max-1)
        self.posterior1 = [self._alpha_(t)*self._beta_(t) for t in range(self.t_max)]
        self.posterior2 = [np.outer(self._alpha_(t-1), self._beta_(t)*self._e[t,:]/self.memo[('c',t)]) * self._a for t in range(1, self.t_max)]
        eps = min(self.eps, 1e-12)
        Q =  (np.log(self._p+eps)*self.posterior1[0]).sum()
        Q += np.sum([(np.log(self._a+eps)*self.posterior2[t]).sum() for t in range(self.t_max-1)])
        Q += np.sum([(np.log(self._e+eps)*self.posterior1[t]).sum() for t in range(self.t_max)])
        
        is_convergent = False
        if len(self._Q_)>0:
            Q_old = self._Q_[-1]
            if Q - Q_old >=0 and Q - Q_old < self.eps*np.abs(Q_old): is_convergent = True
        self._Q_.append(Q)    

        return is_convergent
    
    def _m_step(self):
        self._p = self.posterior1[0]/self.posterior1[0].sum()
        pool_ = 0.0
        for t in range(1, self.t_max):
            pool_ += self.posterior2[t-1]
        self._a = pool_/pool_.sum(axis=1)[:,None]
        
        assert np.all(self._a.sum(axis=1).round(6) == np.ones(self.K))
        
        if self._e_params["type"] == "normal":
            mu_ = norm_ = sig2 = 0.0
            for t in range(self.t_max):
                mu_ += self.x_[t]*self.posterior1[t]
                sig2 += (self.x_[t]-self._e_params["mean"])**2 * self.posterior1[t]
                norm_ += self.posterior1[t]
            mu_ /= norm_
            self._e_params["mean"] = mu_
            sig2 /= norm_
            self._e_params["variance"] = sig2
        else:
            raise ValueError("distribution not tabulated")
        
        self.memo = {}
    
    def _emission(self, t):
        assert t >= 0 and t < self.t_max
        x = self.x_[t]
        gauss = lambda z, m, s: np.sqrt(1/(2*np.pi*s)) * np.exp(-(z-m)**2/(2*s))
        
        if self._e_params["type"] == "normal":
            for k in range(self.K):
                mu, sig2 = self._e_params["mean"][k], self._e_params["variance"][k]
                self._e[t, k] = gauss(x, mu, sig2)
        else:
            raise ValueError("distribution not tabulated")
        return self._e[t,:]
    
    def __emission(self, x, k):
        
        gauss = lambda z, m, s: np.sqrt(1/(2*np.pi*s)) * np.exp(-(z-m)**2/(2*s))
        res = 0.0
        if self._e_params["type"] == "normal":
            mu, sig2 = self._e_params["mean"][k], self._e_params["variance"][k]
            res = gauss(x, mu, sig2)
        else:
            raise ValueError("distribution not tabulated")
        return res
    
    def _alpha_(self, t):
        
        if ('a', t) in self.memo:
            return self.memo[('a',t)]
        
        assert t >= 0 and t < self.t_max
        
        if t == 0:
            temp = self._p * self._emission(0)
            self.memo[('a',0)] = temp/temp.sum()
            self.memo[('c',t)] = temp.sum()
            return self.memo[('a',0)]
        
        temp = self._emission(t) * (self._alpha_(t-1) @ self._a)
        self.memo[('a', t)] = temp / temp.sum()
        self.memo[('c', t)] = temp.sum()
        return self.memo[('a', t)]
        
    def _beta_(self, t):
        assert t >= 0 and t < self.t_max
        
        if ('b',t) in self.memo:
            return self.memo[('b',t)]
        
        if t == self.t_max - 1:
            self.memo[('b',t)] = np.ones(self.K)
            return self.memo[('b',t)]
        
        if ('c',t+1) not in self.memo:
            _ = self._alpha_(t+1)
        
        self.memo[('b',t)] = self._a @ (self._emission(t+1)*self._beta_(t+1)) / self.memo[('c', t+1)]
        
        return self.memo[('b',t)]
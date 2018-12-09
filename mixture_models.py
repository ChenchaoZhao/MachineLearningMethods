import numpy as np

class DirichletProcessMixtureModel:
    def __init__(self, alpha, n_dim = 2, sig2 = 1, max_itr = 50):
        self.alpha = alpha
        self.n_dim = n_dim
        self.sig2 = sig2
        self.max_itr = max_itr
        
    def fit(self, data_ = None, alpha = None, init_method = 0):
        if data_ is None:
            data_ = self.data_
        else:
            self.data_ = data_
        
        if alpha is not None:
            self.alpha = alpha
        
        self.m_sam, self.n_dim = data_.shape
        if init_method == 0:
            self.labels_ = np.zeros(self.m_sam, dtype=int)
        elif init_method == 1:
            self.labels_ = np.arange(self.m_sam)
        
        self.unique_labels_ = np.unique(self.labels_)
        stop_itr = False
        for itr in range(self.max_itr):
            if stop_itr:
                print("Iterations {}".format(itr+1))
                break
            elif itr == self.max_itr - 1:
                print("MaxIterations {} reached".format(itr+1))
            stop_itr = True
            for idx in range(self.m_sam):
                is_changed = self._gibbs_sampler(idx)
                if is_changed:
                    stop_itr = False
        
    
    def _normal_pdf(self, x, mean, sig2=None):
        if sig2 is None:
            sig2 = self.sig2
        return np.exp( - np.sum((x - mean)**2) / 2 / sig2)
    
    def _gibbs_sampler(self, idx):
        # new cluster likelihood
        CRP = self.alpha / (self.m_sam + self.alpha - 1)
        density_likelihood = self._normal_pdf(self.data_[idx,:], 0)
        current_max = CRP * density_likelihood
        current_arg = self.unique_labels_.max() + 1 # new cluster label
        
        # existing clusters without point idx
        for l in self.unique_labels_:
            CRP = np.sum((self.labels_ == l)&(np.arange(self.m_sam) != idx))
            
            points_in_class_l_ = self.data_[np.where((self.labels_ == l)&(np.arange(self.m_sam) != idx))[0],:]
            if points_in_class_l_.shape[0] == 0:
                total_likelihood = 0
            else:
                mean = np.mean(points_in_class_l_, axis=0) * CRP / (CRP+1)
                density_likelihood = self._normal_pdf(self.data_[idx,:], mean)
                total_likelihood = CRP / (self.m_sam + self.alpha - 1) * density_likelihood
            
            if total_likelihood > current_max:
                current_max = total_likelihood
                current_arg = l
#             elif total_likelihood == current_max:
#                 current_arg = min(l, current_arg)
        label_is_changed = (self.labels_[idx] != current_arg)
        self.labels_[idx] = current_arg # set the class label of idx to the most likely class
        self.unique_labels_ = np.unique(self.labels_)
        for jdx in range(self.unique_labels_.size):
            self.labels_[np.where(self.labels_ == self.unique_labels_[jdx])] = jdx
        
        return label_is_changed
    
    def _generate_dirichlet_process_path(self, n_stick_max = 100):
        # stick breaking v iid ~ Beta(1, alpha)
        v_ = np.random.beta(1, self.alpha, size=n_stick_max)
        mu_ = np.random.multivariate_normal([0]*self.n_dim, np.eye(self.n_dim), size=n_stick_max)
        p_ = []
        remain = 1
        for idx in range(n_stick_max):
            p_.append(v_[idx]*remain)
            remain -= p_[-1]
            
        p_ = np.array(p_)
        p_ = p_/p_.sum()
        print("remaining stick {}".format(remain))
        self.DP_path = (p_, mu_)
        return self.DP_path
    def _sample_cluster_parameter_from_DP_path(self, n_samples):
        idx_ = np.arange(len(self.DP_path[0]))
        jdx_ = np.random.choice(idx_, p=self.DP_path[0], replace=True, size=n_samples)
        self.cluster_parameters_ = self.DP_path[1][jdx_,:]
        return self.cluster_parameters_
    
    def _sample_data_from_cluster_parameter(self):
        n_samples = self.cluster_parameters_.shape[0]
        self.simulated_data_ = np.zeros((n_samples, self.n_dim))
        for idx in range(n_samples):
            self.simulated_data_[idx, :] = np.random.multivariate_normal(
                self.cluster_parameters_[idx,:], 
                self.sig2*np.eye(self.n_dim))
        return self.simulated_data_
    def generate_simulated_data(self, n_samples, alpha=None, n_dim=None):
        if alpha is not None:
            self.alpha = alpha
        if n_dim is not None:
            self.n_dim = n_dim
            
        _ = self._generate_dirichlet_process_path()
        _ = self._sample_cluster_parameter_from_DP_path(n_samples)
        self.data_ = self._sample_data_from_cluster_parameter()
        
        return self.data_
    
    def _chinese_resturant_process_distribution(self, idx):
        self.CRP_ = dict.fromkeys(self.unique_labels_, 0)
        for l in self.unique_labels_:
            self.CRP_[l] = np.sum(self.labels_ == l)
            if self.labels_[idx] == l:
                self.CRP_[l] = max(0, self.CRP_[l] - 1)
            self.CRP_[l] /= self.m_sam + self.alpha - 1
        self.CRP_[-1] = self.alpha / (self.m_sam + self.alpha - 1)   
        return self.CRP_
    
    
        
import numpy as np
import scipy as sp
from scipy.sparse.csgraph import laplacian as sp_lap
from sklearn.cluster import KMeans
import math

pi = np.pi
palettes = [
    '#D870AD', '#B377D9', '#7277D5', 
    '#4B8CDC', '#3BB1D9', '#3BBEB0', 
    '#3BB85D', '#82C250', '#B0C151', 
    '#F5BA42', '#F59B43', '#E7663F', 
    '#D94C42', '#655D56', '#A2A2A2']
cc_ = np.array(palettes)


class GegenbauerPolynomial:
    """
    TODO
    """
    
    def __init__(self, alpha, order):
        self.l = order
        self.alpha = alpha
        self.poly = None
        self.memo = {}
        #self.K_1 = None
        
    def get_poly1d(self, order=None):
        if order is not None:
            self.l = order
        return self._gegenbauer(self.l)
    
    def _gegenbauer(self, l):
        if l in self.memo:
            return self.memo[l]
        
        if l < 0:
            raise ValueError("Order should be non-negative but l = {}".format(l))
        
        if l == 0:
            self.memo[l] = np.poly1d([1])
            return self.memo[l]
        if l == 1:
            self.memo[l] = np.poly1d([2*self.alpha, 0])
            return self.memo[l]
        
        self.memo[l] = (2*(l+self.alpha-1)*np.poly1d([1,0])*self._gegenbauer(l-1) 
                       - (l+2*self.alpha-2)*self._gegenbauer(l-2))/l
        
        return self.memo[l]
        
    
    def __call__(self, x, order=None, factor=1):
        if order is not None:
            self.l = order
        poly = self.get_poly1d()*factor
        return poly(x)

class HypersphericalHeatKernel:
    """
    TODO
    """
    def __init__(self, t, dimension_euclidean, max_itr = 100, eps_tol = 1e-8, normed=True):
        self.max_itr, self.eps_tol = max_itr, eps_tol
        self.normed = normed # if True, the kernel will be divided by self similarity
        
        if dimension_euclidean < 2:
            raise ValueError("Euclidean dimension should be above 2")
        else:
            self.dim = dimension_euclidean if dimension_euclidean > 2 else dimension_euclidean + eps_tol
        
        self._effective_time = np.log(self.dim)/self.dim
        self.t = t*self._effective_time # assume t is order 1
        self._t = t
        
        self._p_gegenbauer = GegenbauerPolynomial(self.dim/2 - 1, 0)
        
    def _hyperspherical_heat_kernel(self, x):
        pos = np.zeros_like(x, dtype=np.float64)
        neg = np.zeros_like(x, dtype=np.float64)
        res = np.zeros_like(x, dtype=np.float64)
        r = p = n = 0.0
        d = self.dim
        t = self.t
        
        for l in range(self.max_itr):
            w_ = np.exp(-l*(l+d-2)*t)*(2*l + d - 2)
            
            poly_ = self._p_gegenbauer(x, l, w_) # w_ as the prefactor, x is a vector
            temp = self._p_gegenbauer(1, l, w_)
            if temp >= 0:
                p += temp
            else:
                n += temp
            
            pos[poly_>=0] += poly_[poly_>=0]
            neg[poly_< 0] += poly_[poly_< 0]
            
            res = pos + neg
            r = p + n
            if np.abs(temp) < np.abs(r)*self.eps_tol: # use origin as stopping criterion
                print("Summation terminated at l={}".format(l))
                break
        
        res = 0*(res<0) + res*(res>=0) # near south pole the convergence is slow, so we manually kill the negative terms
        res = np.nan_to_num( res )
        if self.normed:
            res /= r
        
        return res
    
    def __call__(self, cos, t=None):
        if t is not None:
            self.t = t*self._effective_time
        return self._hyperspherical_heat_kernel(cos)

class EffectiveDissimilarity:
    """
    TODO
    """
    def __init__(self, data, metric="euc"):
        if metric == "euc":
            self.effective_dissimilarity_ = {0: self._euc_dist_mat_(data)}
        elif metric == "cos":
            self.effective_dissimilarity_ = {0: self._cos_dist_mat_(data)}
        elif metric == "precomputed":
            if np.sum(data - data.T) != 0 or data.ndim != 2:
                raise ValueError("Input data should be a precomputed distance matrix.")
            self.effective_dissimilarity_ = {0: data}
    
    def get_effective_dissimilarity(self, tau=0):
        
        if tau in self.effective_dissimilarity_:
            return self.effective_dissimilarity_[tau]
        if tau <= 0:
            return self.effective_dissimilarity_[0]
        
        self.effective_dissimilarity_[tau] = self._cos_dist_mat_(self.get_effective_dissimilarity(tau-1))
        
        return self.effective_dissimilarity_[tau]
        
    def _euc_dist_mat_(self, X):
        """
        Data should be in the shape [n_features, n_samples]
        """
        temp = np.dot(X.T, X)
        d_ = np.diag(temp)
        d_1_ = np.outer(d_, np.ones_like(d_))
        d_ij_ = np.sqrt(d_1_ + d_1_.T - 2*temp)
        d_ij_[d_ij_ < 0] = 0
        d_ij_ = (d_ij_+d_ij_.T)/2 # symmetric
        np.fill_diagonal(d_ij_, 0) # zero diagonal     
        return d_ij_
    
    def _cos_dist_mat_(self, D):
        norm_ = D.sum(axis=0)
        X = np.sqrt(np.abs(D/norm_[:,None]))
        d_ij_ = 1-np.dot(X, X.T)
        d_ij_[d_ij_ < 0] = 0
        d_ij_ = (d_ij_+d_ij_.T)/2 # symmetric
        np.fill_diagonal(d_ij_, 0) # zero diagonal
        
        return d_ij_
    
    def get_MDS(self, tau, max_embedding_dimension=3):
        D = self.get_effective_dissimilarity(tau)
        mds = MDS(D, max_embedding_dimension)
        return mds

class MDS:
    """
    TODO
    """
    def __init__(self, distance_matrix, max_embedding_dimension=3):
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix should be a symmetric square matrix")
        self.distance_matrix = distance_matrix
        self.n_sample = distance_matrix.shape[0]
        self.max_embedding_dimension = min(max_embedding_dimension, distance_matrix.shape[0])
        
    def get_embedding(self):
        J = np.eye(self.n_sample) - np.ones([self.n_sample]*2)/self.n_sample
        B = - 0.5 * J @ (self.distance_matrix**2) @ J
        eigval, eigvec = np.linalg.eigh(B)
        self._eigval = eigval[::-1]
        self._eigvec = eigvec[:,::-1]
        self.n_positive = np.sum(eigval > 0)
        self.max_embedding_dimension = min(self.n_positive, self.max_embedding_dimension)
        v_ = self._eigvec[:,:self.max_embedding_dimension]
        l_ = np.sqrt(self._eigval[:self.max_embedding_dimension])
        self.embedding_ = v_*l_
        return self.embedding_

class KernelMDS:
    def __init__(self, kernel_matrix, max_embedding_dimension=3):
        if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
            raise ValueError("Kernel matrix should be a symmetric square matrix")
        self.kernel_matrix = kernel_matrix
        self.n_sample = kernel_matrix.shape[0]
        D2 = self._get_distance_squared(self.kernel_matrix)
        self.distance_squared = D2
        
        self.max_embedding_dimension = min(max_embedding_dimension, self.n_sample)
        
    def _get_distance_squared(self, K):
        """
        D2_ij = < Fi - Fj, Fi - Fj > = K_ii + K_jj - 2 K_ij = (k_i + k_j) delta_ij - 2 K_ij
        """
        k = K.diagonal()
        D2 = np.outer(k, np.ones(self.n_sample)) + np.outer(np.ones(self.n_sample), k) - 2*K
        D2 = (D2 + D2.T)/2
        np.fill_diagonal(D2, 0)
        return D2
    
    def get_embedding(self):
        J = np.eye(self.n_sample) - np.ones([self.n_sample]*2)/self.n_sample
        B = - 0.5 * J @ self.distance_squared @ J
        eigval, eigvec = np.linalg.eigh(B)
        self._eigval = eigval[::-1]
        self._eigvec = eigvec[:,::-1]
        self.n_positive = np.sum(eigval > 0)
        self.max_embedding_dimension = min(self.n_positive, self.max_embedding_dimension)
        v_ = self._eigvec[:,:self.max_embedding_dimension]
        l_ = np.sqrt(self._eigval[:self.max_embedding_dimension])
        self.embedding_ = v_*l_
        return self.embedding_
    
class UndirectedGraph:
    """
    TODO
    """
    
    def __init__(self, data_, graph_embedded=True, eps_quant = 1, normed=True):
        
        self.graph_embedded = graph_embedded
        self.norm = normed
        self.laplacian = None
        self.n_node = data_.shape[0]
        print("> n_node is {}".format(self.n_node))
        self._eigval, self._eigvec = None, None
        self.propagator = None
        
        if graph_embedded:
            self.distance = data_
            if eps_quant > 0 and eps_quant <= 100:
                self.eps_quant = eps_quant
                self.adjacency = self._gaussian_rbf(self.distance)
            else:
                raise ValueError('eps_quant has to be in the range (0,100].')
            print('> Gaussian affinity eps quantile (%): eps_quant = {}'.format(self.eps_quant))
        
        else: 
            # not embedded in Euclidean space, then data is network adjacency
            if data_.shape[0] == data_.shape[1]:
                if np.any(data_ < 0):
                    raise ValueError('Entries of adjacency matrix should be non-negative.')
                self.adjacency = (data_ + data_.T)/2
            else:
                raise ValueError('Adjacency matrix should be a symmetric matrix.')
                
            print('> Initial parameters: graph is not embedded in Euclidean space')
    
    def _gaussian_rbf(self, distance_matrix):        
        D_ = distance_matrix
        D_tri = np.triu(D_, 1)
        dist_ = np.sort(D_tri[D_tri>0])
        r_eps = np.percentile(dist_, self.eps_quant)
        self.r_eps = r_eps
        return np.exp(- D_**2 / r_eps**2)
    
    def _adj_to_laplacian(self, adjacency):
        return sp_lap(adjacency, normed=self.norm, return_diag=False)
        
    def get_laplacian(self):
        A = self._gaussian_rbf(self.distance) if self.graph_embedded else self.adjacency
        L = self._adj_to_laplacian(A)
        self.laplacian = L
        return L
    
    def get_spectrum(self):
        L = self.get_laplacian() if self.laplacian is None else self.laplacian
        self._eigval, self._eigvec = np.linalg.eigh(L)
        self._eigval = self._eigval - self._eigval[0]
        return self._eigval, self._eigvec
    
    def get_propagator(self, mass2=None):
        
        if self._eigval is None:
            _, _ = get_spectrum()
        
        if mass2 is None:
            mass2 = np.max(np.diff(self._eigval))
        
        H = self.laplacian
        G = np.linalg.inv(H + mass2*np.eye(self.n_node))
        self.propagator = G
        
        return G
    
    def get_embedding(self, which='G', max_embedding_dimension=3):
        if which == 'G':
            return KernelMDS(self.propagator, max_embedding_dimension)
        elif which == 'A':
            return KernelMDS(self.adjacency, max_embedding_dimension)

        

class SpectralClustering:
    """
    
    The class SpectralClustering is initialized by specifying the number of clusters: n_cluster
    
    Other optional parameters:
        - norm_method: None, "row" or "deg". Default is "row."
            If None, spectral embedding is not normalized;
            If "row," spectral embedding is normalized each row (each row represent a node);
            If "deg," spectral embedding is normalized by degree vector.
        - is_exact: bool. Default is True.
            If True, exact eigenvectors and eigenvalues will be computed.
            If False, first n_cluster low energy eigenvectors and eigenvalues (small eigenvalues) will be computed
    
    Method:
    
    fit (Laplacian_matrix) compute eigenvalue and eigenvectors of Laplacian_matrix and perform spectral embedding.
        
        >> clf = SpectralClustering(n_cluster=5)
        >> clf.fit(Laplacian)
    
    Attribute:
    
    labels_, a numpy array containing the class labels, e.g.
    
        >> clf.labels_
    
    Reference
    
    A Tutorial on Spectral Clustering, 2007 Ulrike von Luxburg http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    
    """
    
    def __init__(self, n_clusters, norm_method='row', is_exact=True):
        self.nc = n_clusters
        self.is_exact = is_exact
        if norm_method == None:
            self.norm = 0 # do not normalize
        elif norm_method == 'row':
            self.norm = 1 # row normalize
        elif norm_method == 'deg':
            self.norm = 2 # deg normalize
        else:
            raise ValueError("norm_method can only be one of {None, 'row', 'deg'}.")
        
        print('> Initialization parameters: n_cluster={}'.format(self.nc))
        if self.norm == 0:
            print('> Unnormalized spectral embedding.')
        elif self.norm == 1:
            print('> Row normalized spectral embedding.')
        elif self.norm == 2:
            print('> Degree normalized spectral embedding.')

    def compute_eigs(self):
        if self.is_exact:
            self.Heigval, self.Heigvec = np.linalg.eigh(self.H_)
            print('> Exact eigs done')
        else:
            self.Heigval, self.Heigvec = sp.sparse.linalg.eigsh(self.H_, k=self.nc, which='SM')
            print('> Approximate eigs done')
    
    def fit(self, Lap_):
        
        self.H_ = Lap_
        self.compute_eigs()
        
        if self.norm == 0:
            spec_embed = self.Heigvec[:,:self.nc]
        elif self.norm == 1:
            spec_embed = (self.Heigvec[:,:self.nc].T/np.sqrt(np.sum(self.Heigvec[:,:self.nc]**2, axis=1))).T
        elif self.norm == 2:
            spec_embed = (self.Heigvec[:,:self.nc].T / np.abs(self.Heigvec[:,0])).T
        
        km_ = KMeans(n_clusters = self.nc, n_init=100).fit(spec_embed)
        self.labels_ = km_.labels_     
        
class QuantumTransportClustering:
    """
    
    The Class QuantumTransportClustering is initialized by specifying:
        - n_cluster: the number clusters
        - Hamiltonian: a symmetric graph Laplacian
    
    Other optional paramters:
        - s = 1.0: used to generate the Laplace transform s-parameter, 
          which is s * (E[n_cluster - 1]/(n_cluster - 1)).
        - is_exact = True: if True, exact eigenvalue and eigenvectors 
          of graph Laplacian will be computed
        - n_eigs = None: If is_exact = False, first n_eigs low energy 
          eigenvectors and eigenvalues will be computed, by default 
          n_eigs = min(10*n_clusters, number_of_nodes). If is_exact 
          = True, but n_eigs is not None, then first n_eigs low energy
          exact eigenstates will be used in QT clustering.
    
    Methods:
    
        - Grind():
          Generate quantum transport with a set of initialization nodes and extract their phase distributions.
          The phase distributions are mapped to a set of class label vectors ranging from 0 to n_cluster-1
          The class label vectors are stored in matrix "Omega_"
    
        - Espresso()
          Apply "direct extraction method" to raw labels Omega_
          The final decision can be obtained using "labels_"
    
        - Coldbrew()
          Apply "consensus matrix method" to raw labels Omega_
          The consensus matrix can be obtained using "consensus_matrix_"
    
    Attributes:
    
        - Omega_, the Omega matrix whose columns are class labels associated with initialization nodes
    
        - labels_, the predicted class labels using direct extraction or "Espresso()" method
    
        - consensus_matrix_, the consensus matrix based on Omega matrix using "Coldbrew()" method
    
    """
    
    machine_eps = np.finfo(float).eps
    
    def __init__(self, n_clusters, Hamiltonian, s=1.0, is_exact=True, n_eigs = None):
        
        if Hamiltonian.shape[0] == Hamiltonian.shape[1]:
            self.m_nodes = Hamiltonian.shape[0]
            n_primes = self.m_nodes
            self.H_ = Hamiltonian
        else:
            raise ValueError('Hamiltonian or Graph Laplacian should be a symmetric matrix.')
    
    
        if isinstance(n_clusters, int) and n_clusters > 1:
            self.nc = n_clusters
            print('> Initialization parameters: n_cluster={}'.format(self.nc))
        else:
            raise ValueError('Number of clusters is an int and n_clusters > 1.')
    
        self.is_exact = is_exact
        if not is_exact:
            self._n_eigs = min(10*n_clusters, self.m_nodes) if n_eigs == None else n_eigs
            print('> First {} low energy eigenvalues will be computed.'.format(self._n_eigs))
    
        self.n_eigs = n_eigs
    
        self.s = s
        print('> Laplace variable s = {}'.format(self.s))
    
        self.primes_ = self.generate_primes(n_primes)
    
        print('> First {} primes generated'.format(n_primes))     
        print('>> Espresso: direct extraction method')
        print('>> Coldbrew: consensus matrix method')
            
        

    def compute_eigs(self):
        if self.is_exact:
            self.Heigval, self.Heigvec = np.linalg.eigh(self.H_)
            print('> Exact eigs done')
        else:
            self.Heigval, self.Heigvec = sp.sparse.linalg.eigsh(self.H_, k=self._n_eigs, which='SM')
            print('> Approximate eigs done')
        
        gaps_ = np.abs(np.diff(self.Heigval))
        if np.any(gaps_ < self.machine_eps):
            print('> Warning: Some energy gaps are smaller than machine epsilon. QTC results may show numerial instability.')
    
    def generate_primes(self, n):
        """
        Find first n primes
        """
        if n <= 0:
            raise ValueError('n_prime is int and > 0.')
        elif n == 1:
            return [1]
        elif n == 2:
            return [1,2]
        elif n == 3:
            return [1,2,3]
        else:
            primes_ = [1,2,3]
            itr = n - 3

        new = primes_[-1]

        while itr > 0:
            new += 1
            is_prime = False

            if new & 1 == 1:
                is_prime = True
                for k in range(3, int(math.floor(math.sqrt(new)))+1):
                    if new % k == 0:
                        is_prime = False
                        break
            if is_prime:
                primes_.append(new)
                itr -= 1
        return np.array(primes_)
    
    def laplace_transform_wf_(self, s, eigval_, eigvec_, init_vec_):
        # expand init_vec_ in terms of eig_vec_
        coeff_ = np.dot(init_vec_, eigvec_)
        w_ = coeff_/(s+1j*eigval_)
        psi_s_ = np.dot(eigvec_, w_)
        return psi_s_
    
    def phase_info_clustering_diff_(self, phase_, n_cluster=None):
        if n_cluster == None:
            n_cluster = self.nc
        elif n_cluster == 1:
            class_label_ = np.zeros(phase_.size)
        else:
            while np.any(phase_ < 0):
                phase_[phase_<0] += 2*pi
            while np.any(phase_ > 2*pi):
                phase_[phase_>2*pi] -= 2*pi   
            idx_ = np.argsort(phase_)
            iidx_ = np.argsort(idx_)
            z_ = np.exp(1j*phase_[idx_])
            diff_ = np.zeros(z_.size, dtype=float)
            diff_[0] = np.abs(z_[0] - z_[-1])
            diff_[1:] = np.abs(np.diff(z_))
            n_part = n_cluster
            partition_idx_ = np.argpartition(diff_, -n_part)[-n_part:]
            partition_idx_ = np.sort(partition_idx_)
            class_label_ = np.zeros_like(idx_)
            for k in range(1,n_cluster):
                class_label_[partition_idx_[k-1]:partition_idx_[k]] += k
            class_label_ = class_label_[iidx_]
        return class_label_
    
    def phase_info_clustering_KMeans_(self, phase_, n_cluster):
        if n_cluster == None:
            n_cluster = self.nc
        elif n_cluster == 1:
            class_label_ = np.zeros(phase_.size)
        else:
            z_ = np.exp(1j*phase_)
            data_ = np.vstack((z_.real, z_.imag)).T
            km = KMeans(n_clusters=n_cluster)
            km.fit(data_)
            class_label_ = km.labels_
        return class_label_
    
    def Grind(self, s=None, grind='medium', method='diff', init_nodes_=None):
        """
        grind option can be "coarse", "medium", "fine", "micro", "custom"
        If grind="custom" one need to specify the a list of nodes as init_nodes_
        """
        
        if s == None:
            s = self.s
        else:
            self.s = s
            print('> Update: Laplace variable s = {}'.format(s))
        
        
        if grind == 'coarse':
            _every_ = self.m_nodes // 30
        elif grind == 'medium':
            _every_ = self.m_nodes // 60
        elif grind == 'fine':
            _every_ = self.m_nodes // 100
        elif grind == 'micro':
            _every_ = 1
        elif grind == 'custom':
            if init_nodes_ == None:
                raise ValueError('> If grind is custom, you need to specify a list of initialization nodes, e.g. init_nodes_=[0,1,2,3,10]')
            else:
                init_nodes_ = np.array(init_nodes_)
                _every_ = None
        else:
            raise ValueError('> parameter grind can be {coarse, medium, fine, micro, or custom}.')
        
        self.compute_eigs() # compute Heigval and Heigvec
        
        deg_idx_ = np.argsort(self.Heigvec[:,0]**2)
        
        if _every_ != None:
            if _every_ == 0: _every_ = 1
            init_nodes_ = deg_idx_[::_every_]
        
        m_init = init_nodes_.size
        self.m_init = m_init
        print('> {}-ground: {} initialization nodes'.format(grind, m_init))
        
        Omega_ = np.zeros((self.m_nodes, m_init), dtype=int) # col for init_
        
        show_warning = False
        
        for jdx, idx in enumerate(init_nodes_):
            init_ = np.zeros(self.m_nodes)
            init_[idx] += 1.0
            psi_s_ = self.laplace_transform_wf_(s*(self.Heigval[self.nc-1]-self.Heigval[0])/(self.nc-1), \
            self.Heigval[:self.n_eigs]-self.Heigval[0], self.Heigvec[:,:self.n_eigs], init_)
            rho_s_ = np.abs(psi_s_)
            theta_s_ = np.angle(psi_s_)
            
            if np.any(rho_s_ < self.machine_eps):
                show_warning = True
#                 theta_s_[np.where(psi_s_ < self.machine_eps)] = 0.0
#                 print('! Small amplitude warning !')
            
            if method == 'diff':
                Omega_[:,jdx] = self.phase_info_clustering_diff_(theta_s_, self.nc)
            elif method == 'kmeans':
                Omega_[:,jdx] = self.phase_info_clustering_KMeans_(theta_s_, self.nc)
            else:
                raise ValueError('> method can only be {diff, or kmeans}.')

        self.Omega_ = Omega_
        if show_warning:
            print('>> Warning: some amplitudes are below machine eps. QTC may show numerical instability.')
        return Omega_
    
    def is_equiv(self, Omg_cols_):
    
        p_ = np.sqrt(self.primes_) 

        unique_col_ = np.unique(Omg_cols_, axis=1, return_counts=False)
        n_cluster = self.nc

        mix_all_ = unique_col_ @ p_[:unique_col_.shape[1]]
        if np.unique(mix_all_).size == n_cluster:
            return True
        else:
            return False
    
    def pigeonhole(self, columns_, columns_idx=None, hashtab=None):
    
        if columns_idx == None:
            columns_idx = list(range(columns_.shape[1]))

        n_cols_ = len(columns_idx)

        if hashtab == None:
            hashtab = dict()


        if self.is_equiv(columns_[:,columns_idx]):
            is_existing = False
            for hashtag in hashtab.keys():
                if self.is_equiv(columns_[:,[columns_idx[0],hashtag]]):
                    hashtab[hashtag] += columns_idx
                    is_existing = True

            if not is_existing:
                new_hashtag = columns_idx[0]            
                hashtab[new_hashtag] = columns_idx
        else:
            columns_idx_1 = columns_idx[:n_cols_//2]
            columns_idx_2 = columns_idx[n_cols_//2:]        

            hashtab = self.pigeonhole(columns_, columns_idx_1, hashtab)
            hashtab = self.pigeonhole(columns_, columns_idx_2, hashtab)

        return hashtab
        
    
    def Espresso(self):
        Table_ = self.pigeonhole(self.Omega_)
        
        alpha_ = np.array(list(Table_.keys()))
        weight_alpha_ = np.zeros(alpha_.size)
        
        for j in range(alpha_.size):
            weight_alpha_[j] = len(Table_[alpha_[j]])
            
        sort_idx_ = np.argsort(weight_alpha_)[::-1]
        weight_alpha_ = weight_alpha_[sort_idx_]
        weight_alpha_ /=  weight_alpha_.sum()
        alpha_ = alpha_[sort_idx_]
        
        self.double_shot_ = (alpha_, weight_alpha_)
        self.labels_ = self.Omega_[:,alpha_[0]]
        
    def Coldbrew(self):
        C_ = np.diag([self.m_init]*self.m_nodes)
        for idx in range(self.m_nodes):
            for jdx in range(idx+1, self.m_nodes):
                temp = np.sum(self.Omega_[idx,:] == self.Omega_[jdx, :])
                C_[idx, jdx] = temp
                C_[jdx, idx] = temp
                
        self.consensus_matrix_ = C_



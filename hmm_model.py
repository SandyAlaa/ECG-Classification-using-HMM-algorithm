"""
Custom Gaussian Hidden Markov Model — implemented from scratch using NumPy/SciPy.

Each class has its own HMM. Classification is done by scoring a sequence
under every class HMM and picking the highest log-likelihood.
"""

import numpy as np
from scipy.special import logsumexp


class GaussianHMM:
    """
    Gaussian HMM with diagonal covariance matrices.

    Training  : Baum-Welch (EM) algorithm
    Decoding  : Viterbi algorithm
    Scoring   : Forward algorithm (log-likelihood)
    """

    def __init__(self, n_states: int = 4, n_iter: int = 60, tol: float = 1e-4,
                 random_state: int = 42):
        self.n_states    = n_states
        self.n_iter      = n_iter
        self.tol         = tol
        self.random_state = random_state
        self.is_fitted   = False

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #
    def _init_params(self, all_obs: np.ndarray, n_features: int):
        rng = np.random.RandomState(self.random_state)
        K   = self.n_states

        # --- initial state distribution (uniform) ---
        self.log_pi = np.full(K, -np.log(K))

        # --- transition matrix (left-to-right bias) ---
        A = np.zeros((K, K))
        for i in range(K):
            A[i, i]            = 0.6
            A[i, (i + 1) % K] = 0.4
        A /= A.sum(axis=1, keepdims=True)
        self.log_A = np.log(A + 1e-300)

        # --- emission parameters (k-means++ style initialisation) ---
        n = len(all_obs)
        first = rng.randint(n)
        centres = [all_obs[first]]
        for _ in range(K - 1):
            dists = np.array([min(np.sum((x - c)**2) for c in centres)
                              for x in all_obs])
            probs = dists / dists.sum()
            centres.append(all_obs[rng.choice(n, p=probs)])

        self.mu    = np.array(centres)                          # (K, D)
        self.sigma = np.tile(np.var(all_obs, axis=0) + 1e-6,
                             (K, 1))                            # (K, D)

    # ------------------------------------------------------------------ #
    #  Emission log-probability
    # ------------------------------------------------------------------ #
    def _log_emission_matrix(self, obs: np.ndarray) -> np.ndarray:
        """
        obs  : (T, D)
        return: (T, K)  —  log P(o_t | state k)
        """
        T, D = obs.shape
        K    = self.n_states
        log_b = np.zeros((T, K))
        for k in range(K):
            diff        = obs - self.mu[k]             # (T, D)
            log_b[:, k] = -0.5 * np.sum(
                diff**2 / self.sigma[k] + np.log(2 * np.pi * self.sigma[k]),
                axis=1
            )
        return log_b                                    # (T, K)

    # ------------------------------------------------------------------ #
    #  Forward / Backward
    # ------------------------------------------------------------------ #
    def _forward(self, obs: np.ndarray):
        log_b     = self._log_emission_matrix(obs)     # (T, K)
        T, K      = log_b.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = self.log_pi + log_b[0]
        for t in range(1, T):
            log_alpha[t] = (logsumexp(log_alpha[t-1] + self.log_A.T,
                                      axis=1) + log_b[t])
        return log_alpha, log_b

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        T, K      = log_b.shape
        log_beta  = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                self.log_A + log_b[t+1] + log_beta[t+1], axis=1
            )
        return log_beta

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def score(self, obs: np.ndarray) -> float:
        """Log-likelihood of the observation sequence."""
        log_alpha, _ = self._forward(obs)
        return float(logsumexp(log_alpha[-1]))

    def predict_states(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi decoding — returns most-likely state sequence."""
        log_b    = self._log_emission_matrix(obs)
        T, K     = log_b.shape
        viterbi  = np.empty((T, K))
        psi      = np.zeros((T, K), dtype=int)

        viterbi[0] = self.log_pi + log_b[0]
        for t in range(1, T):
            trans      = viterbi[t-1] + self.log_A.T    # (K, K) → column j
            psi[t]     = np.argmax(trans, axis=1)
            viterbi[t] = np.max(trans, axis=1) + log_b[t]

        # back-track
        states    = np.empty(T, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def fit(self, sequences: list, verbose: bool = False):
        """
        Baum-Welch EM on a list of observation sequences.
        Each sequence has shape (T, D).
        """
        n_features = sequences[0].shape[1]
        all_obs    = np.vstack(sequences)
        self._init_params(all_obs, n_features)

        K       = self.n_states
        prev_ll = -np.inf
        ll_history = []

        for iteration in range(self.n_iter):
            # accumulators
            acc_pi   = np.zeros(K)
            acc_A    = np.zeros((K, K))
            acc_mu   = np.zeros((K, n_features))
            acc_mu2  = np.zeros((K, n_features))
            acc_occ  = np.zeros(K)
            total_ll = 0.0

            for obs in sequences:
                T = len(obs)
                log_alpha, log_b = self._forward(obs)
                log_beta         = self._backward(log_b)

                # sequence log-likelihood
                ll        = float(logsumexp(log_alpha[-1]))
                total_ll += ll

                # --- gamma: (T, K) ---
                log_gamma  = log_alpha + log_beta
                log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
                gamma      = np.exp(log_gamma)

                acc_pi  += gamma[0]
                acc_occ += gamma.sum(axis=0)

                # weighted stats for emission update
                for k in range(K):
                    w          = gamma[:, k]          # (T,)
                    acc_mu[k]  += (w[:, None] * obs).sum(axis=0)
                    acc_mu2[k] += (w[:, None] * obs**2).sum(axis=0)

                # --- xi: sum over time ---
                for t in range(T - 1):
                    log_xi  = (log_alpha[t, :, None]
                               + self.log_A
                               + log_b[t+1, None, :]
                               + log_beta[t+1, None, :])
                    log_xi -= logsumexp(log_xi)
                    acc_A  += np.exp(log_xi)

            # ---- M-step ----
            self.log_pi = np.log(acc_pi / acc_pi.sum() + 1e-300)

            row_sum      = acc_A.sum(axis=1, keepdims=True)
            self.log_A   = np.log(acc_A / (row_sum + 1e-300) + 1e-300)

            for k in range(K):
                if acc_occ[k] > 1e-10:
                    self.mu[k]    = acc_mu[k] / acc_occ[k]
                    self.sigma[k] = (acc_mu2[k] / acc_occ[k]
                                     - self.mu[k]**2 + 1e-6)
                    self.sigma[k] = np.maximum(self.sigma[k], 1e-6)

            ll_history.append(total_ll)
            if verbose:
                print(f"  iter {iteration+1:3d}: log-likelihood = {total_ll:.4f}")

            if abs(total_ll - prev_ll) < self.tol:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}")
                break
            prev_ll = total_ll

        self.log_likelihood_ = total_ll
        self.ll_history_     = ll_history
        self.is_fitted       = True
        return self


# ------------------------------------------------------------------ #
#  Multi-class HMM Classifier
# ------------------------------------------------------------------ #
class HMMClassifier:
    """
    One GaussianHMM per class. Classification = argmax log-likelihood.
    """

    def __init__(self, n_states: int = 4, n_iter: int = 60, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.models_  = {}
        self.classes_ = []
        self.is_fitted = False

    def fit(self, X: list, y: np.ndarray, verbose: bool = False):
        """
        X : list of (T, D) observation sequences
        y : class label per sequence
        """
        self.classes_ = sorted(set(y))
        for label in self.classes_:
            seqs = [X[i] for i in range(len(X)) if y[i] == label]
            if verbose:
                print(f"\nTraining HMM for class '{label}' "
                      f"({len(seqs)} sequences) …")
            hmm = GaussianHMM(n_states=self.n_states,
                              n_iter=self.n_iter,
                              tol=self.tol,
                              random_state=hash(label) % (2**31))
            hmm.fit(seqs, verbose=verbose)
            self.models_[label] = hmm
        self.is_fitted = True
        return self

    def predict(self, X: list) -> np.ndarray:
        scores = self._score_matrix(X)
        return np.array([self.classes_[i] for i in np.argmax(scores, axis=1)])

    def predict_proba(self, X: list) -> np.ndarray:
        scores = self._score_matrix(X)
        # Softmax for display probabilities
        scores -= scores.max(axis=1, keepdims=True)
        exp_s   = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def _score_matrix(self, X: list) -> np.ndarray:
        n = len(X)
        K = len(self.classes_)
        mat = np.zeros((n, K))
        for j, label in enumerate(self.classes_):
            mat[:, j] = [self.models_[label].score(x) for x in X]
        return mat
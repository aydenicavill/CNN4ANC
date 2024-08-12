"""EDITER class.

Given a noisy signal S = S* + N and noise data sampled by n external detectors E_1,...,E_n.
EDITER computes the noise free signal by solving:
argmin_{h} |S - Eh| , where E is a block-Toeplitz matrix mapping
the external detector noise data to a window of data points from S.
EDITER works in two phases, it computes an initial matrix of convolution
vectors H, and uses this matrix to establish a clustering of
the shots. Then, a second matrix H2 is computed where one vector h
is optimized for each window of clustered shots.
The clustering has the goal of making the convolution vectors
more robust (by using more data to compute each of them).

For more info about EDITER, the initial paper:
https://onlinelibrary.wiley.com/doi/10.1002/mrm.28992
For information on our changes made to the classic EDITER method to obtain the optimized version:
https://www.notion.so/chipiron/Improving-the-second-part-of-EDITER-line-clustering-9b3f56856db74858bedc8452cd52418d.

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ..conversion.types import TYPE_COMPLEX


# Default EDITER parameters
DEFAULT_DELTA_KX_1 = 7  # first phase convolution window in "first dim" direction.
DEFAULT_DELTA_KX_2 = 12  # second phase convolution window in "first dim" direction.
DEFAULT_DELTA_KY_1 = 0  # first phase convolution window in "second dim" direction.
DEFAULT_DELTA_KY_2 = 0  # second phase convolution window in "second dim" direction.
DEFAULT_CORREL_THRESHOLD = (
    0.75  # correlation threshold for the shot clustering ahead of second phase.
)
# Practical parameters
EDITER_BUFFER_SIZE = 1000  # max size of data (number of shots) to load and process at once.


class EDITER:
    """Class for EDITER method.

    Attributes:
        delta_kx1 (int) : first phase convolution window in "first dim" direction.
        delta_kx2 (int) : second phase convolution window in "first dim" direction.
        delta_ky1 (int) : first phase convolution window in "second dim" direction.
        delta_ky2 (int) : second phase convolution window in "second dim" direction.
        correl_threshold (float) : in [0,1[ threshold for clustering the shots ahead of second phase.
        original_H_mat (array) : H matrix used for shot clustering. Computed via
            unishot_fit or provided as argument to _shot_clustering.
        clusters (list of arrays) : list of indices defining the clusters of shots.
            Computed via _shot_clustering or provided as argument to multishot_fit.

    """

    # Given a noisy signal S = S* + N and noise data sampled by n external detectors E_1, ..., E_n.
    # EDITER computes the noise free signal by solving:
    # argmin_{h} |S - Eh|, where E is a block-Toeplitz matrix contructed from the external detector noise data.
    # here h represents a convolution mapping from the noise signals E_1...E_n to the noise N
    # which pollutes the signal S.
    # The pure signal S* is then computed as as S* = S - Eh.

    def __init__(
        self,
        delta_kx1=DEFAULT_DELTA_KX_1,
        delta_kx2=DEFAULT_DELTA_KX_2,
        threshold=DEFAULT_CORREL_THRESHOLD,
        delta_ky1=DEFAULT_DELTA_KY_1,
        delta_ky2=DEFAULT_DELTA_KY_2,
    ):
        """Initialize EDITER convolution parameters."""
        self.set_convolution_sizes(delta_kx1, delta_kx2, delta_ky1, delta_ky2)
        self.reset_correlation_threshold(threshold)

        # contains the H matrix (of impulse response vestors h) obtained from unishot_fit.
        self.original_H_mat = None
        # contains the list of clusters obtained from _shot_clustering.
        self.clusters = None

    def set_convolution_sizes(self, delta_kx1=None, delta_kx2=None, delta_ky1=None, delta_ky2=None):
        """Set value of convolution sizes without having to contruct a new EDITER class object.

        Args:
            delta_kx1 (int) : first phase convolution window in "first dim" direction.
            delta_kx2 (int) : second phase convolution window in "first dim" direction.
            delta_ky1 (int) : first phase convolution window in "second dim" direction.
            delta_ky2 (int) : second phase convolution window in "second dim" direction.

        Returns:
                None.
        """
        if delta_kx1 is not None:
            self.delta_kx1 = delta_kx1
        if delta_kx2 is not None:
            self.delta_kx2 = delta_kx2
        if delta_ky1 is not None:
            self.delta_ky1 = delta_ky1
        if delta_ky2 is not None:
            self.delta_ky2 = delta_ky2

    def reset_correlation_threshold(self, threshold=DEFAULT_CORREL_THRESHOLD):
        """Reset threshold value used to binarize correlation matrix h.

        Args:
            threshold (float) : new value for the correlation threshold.

        Returns:
            None.
        """
        self.correl_threshold = threshold

    def unishot_fit(self, data_mri, data_noise):
        """Compute first phase corrections by fitting each shot individually.

        Compute first phase corrections, i.e each shot i is treated separately
        to compute its h_i impulse response vector.

        Args:
            data_mri (array) : shape (nb_shots, nb_samples_per_shot)
                raw signal data to be corrected.
            data_noise (array) : shape (nb_shots, nb_samples_per_shot, nb_channels)
                EMI data to use for the correction.

        Returns:
            corrected (array) : corrected signal data.
            H (array) : contains h_i vector of each shot i.
        """
        # FIXME: temporary shape fix until the shape standard is applied in the code
        data_mri = data_mri.T
        data_noise = np.transpose(data_noise, (1, 0, 2))

        # First phase of impulse response vectors computation, each shot is treated as a separate S signal.
        nb_samples_per_shot, nb_shots, nb_channels = data_noise.shape
        delta_kx = self.delta_kx1
        delta_ky = self.delta_ky1

        corrected = np.zeros_like(data_mri)
        convolution_size = nb_channels * (2 * delta_kx + 1) * (2 * delta_ky + 1)
        H = np.zeros(
            (convolution_size, nb_shots), dtype=TYPE_COMPLEX
        )  # H matrix containing impulse response vectors h of each shot.

        # For each shot, create block-Toeplitz noise matrix E and compute h as argmin |Eh - s|
        for shot_idx in range(nb_shots):
            noise_mat = _construct_noise_mat(
                data_noise[:, shot_idx].reshape((nb_samples_per_shot, 1, nb_channels)),
                delta_kx,
                delta_ky,
            )
            E_toeplitz_mat = noise_mat.reshape((convolution_size, -1)).T

            # Reshape raw signal corresponding to the shot number shot_idx:
            sub_signal_mat = data_mri[:, shot_idx].reshape((nb_samples_per_shot, 1))

            # Compute impulse response vector h of this shot by solving argmin |S - Eh| for this shot.
            # h is of size nb_channels * (2 * delta_kx + 1) and represents the "coefficients"
            # of the convolution that computes the mri noise at time t using the convolution
            # window (t - delta_kx, t + delta_kx) from both EMI channels.
            h, _, _, _ = np.linalg.lstsq(E_toeplitz_mat, sub_signal_mat, rcond=None)
            H[:, shot_idx] = h.flatten()

            # lstsq only returns 2-norm of residual (i.e. |S - Eh|²), so we need to compute
            # the corrected signal as S - Eh
            Eh = np.matmul(E_toeplitz_mat, h).flatten()  # constructed noise N for this shot.
            corrected[:, shot_idx] = sub_signal_mat.flatten() - Eh  # S* = S - N

        self.original_H_mat = H  # save H matrix for possible later use (shot clustering)
        # FIXME: temporary fix until the shape standard is applied in the code
        corrected = corrected.T
        return corrected, H



    ################################################ TO DO #################################################
    def _shot_clustering(self, H_mat=None):
        """Computes shot clustering based on h correlation matrix.

        Args:
            H_mat : if different to None, then shot clustering is to be done with
                this user provided matrix rather than the one computed from unishot_fit.

        Returns:
            clusters (array) : list of arrays containing indices of each cluster window.

        """
        # Compute and threshold the impulse response correlation matrix to use for shot clustering
        # normalize H matrix
        if H_mat is not None:
            H = H_mat
        elif self.original_H_mat is not None:
            H = self.original_H_mat
        else:
            raise AttributeError(
                " Shot clustering cannot be done without a H matrix to compute correlations."
                " Please provide it as argument H_mat or call unishot_fit to compute it first."
            )

        threshold = self.correl_threshold
        H_normalized = np.zeros(H.shape, dtype=TYPE_COMPLEX)
        nb_shots = H_normalized.shape[1]

        for shot_idx in range(nb_shots):
            H_normalized[:, shot_idx] = H[:, shot_idx] / np.linalg.norm(H[:, shot_idx])
        
        # Compute Correlation matrix and threshold it to binary matrix
        h_correlation = np.matmul(np.matrix(H_normalized).conj().T, H_normalized)
        h_correlation_mask = (np.abs(h_correlation) > threshold).astype(int)
        for i in range(h_correlation_mask.shape[0]):
            h_correlation_mask[i, i] = 1

        # Arrange shots into windows using the thresholded correlation matrix
        nb_shots = h_correlation_mask.shape[0]
        remaining_shots = np.arange(nb_shots)
        clusters = []
        '''
        ### NEIGHBORHOOD CLUSTERING METHOD ###
        while len(remaining_shots) > 0:
            shot_idx = np.min(remaining_shots)
            # Each cluster includes successive shots which are correlated with the current one (indexed shot_idx).
            # We stop the cluster at the first shot not correlated with the current one.
            if np.where(h_correlation_mask[shot_idx, shot_idx:] == 0)[1].size > 0:
                last_clust_shot = np.min(np.where(h_correlation_mask[shot_idx, shot_idx:] == 0)[1])
            else:
                last_clust_shot = (
                    nb_shots - shot_idx
                )  # if all remaining shots are correlated, then they form the last cluster.
            clusters.append(
                np.arange(shot_idx, shot_idx + last_clust_shot)
            )  # add newly formed cluster of shots
            remaining_shots = np.arange(shot_idx + last_clust_shot, nb_shots)
        '''
        ### GLOBAL CLUSTERING METHOD ###
        ### Creates clusters of all correlated repetitions,
        ### not just ones nearby eachother
        while sum(remaining_shots) != -nb_shots:
            shot_idx = np.min(remaining_shots[remaining_shots!=-1])
            cluster_indices = np.where(h_correlation_mask[shot_idx,:]==1)[1]
            clusters.append(cluster_indices)
            remaining_shots[cluster_indices] = -1

        if len(clusters) > 1:
            ### KMEANS BINNING METHOD ###
            n_clusters = np.arange(2, 9) # how many clusters to consider
            # scores each number of clusters used, uses cluster # with max score
            scores = []
            for n in n_clusters:
                model = KMeans(n_clusters=n, random_state=0)
                labels = model.fit_predict(abs(H_normalized.T))
                score = silhouette_score(abs(H_normalized.T), labels)
                scores.append(score)

            n_clusters = int(np.where(scores==np.max(scores))[0] + 2)
            model = KMeans(n_clusters)
            labels = model.fit_predict(abs(H_normalized.T))

            clusters = []
            for i in range(n_clusters):
                cluster_indices = np.where(labels==(i))[0]
                clusters.append(cluster_indices)

        self.clusters = clusters
        return clusters

    def multishot_fit(
        self,
        data_mri,
        data_noise,
        cluster_list=None,
        optimized=True,
        initial_alphas=None,
        max_iter=20,
        conv_threshold=1e-6,
    ):
        r"""Computes second phase corrections based on clustering of shots.

        Args:
            data_mri (array) : (nb_shots, nb_samples_per_shot)
                raw signal data to be corrected.
            data_noise (array) : (nb_shots, nb_samples_per_shot, nb_channels)
                EMI data to use for the correction.
            cluster_list (list) : if not None, then it is used as the list of shot clusters
                to use.
            optimized (boolean) : whether to use the optimized version of the method,
                i.e. with iterative alpha and h optimization, where the optimization
                problem becomes:
                $\min_{h, \alpha_1,\dots,\alpha_l} \sum_i \| s_i - \alpha_i E_i h \|^2$
            initial_alphas (array) : initial values for alpha optimizition.
            max_iter (int) : maximum iterations for iterative fitting of h and alphas values
                for each window of shots.
            conv_threshold (float) : convergence threshold for iterative fitting of h and alphas
                values foreach window of shots.

        Returns:
            if optimize is False:
                classic_corrected (array) : corrected signal data from classic version of EDITER.
                classic_H_mat (array) : final H matrix.
            if optimize is True:
                optim_corrected (array) : corrected signal data from optimized version of EDITER.
                optim_H_mat (array) : final H matrix.
                alpha_H_mat (array) : final h * alpha matrix.
                final_alphas (array) : final alpha values.

        """
        # Second phase of impulse response vectors computation: shots treated in clustered windows.
        # Check whether the method needs to build the clusters itself via unishot_fit and/or _shot_clustering
        if cluster_list is not None:
            clusters = cluster_list
        elif self.clusters is not None:
            clusters = self.clusters
        elif self.original_H_mat is not None:
            clusters = self._shot_clustering()
        else:
            _, H = self.unishot_fit(data_mri, data_noise)
            clusters = self._shot_clustering(H)

        # FIXME: temporary shape fix until the shape standard is applied in the code
        data_mri = data_mri.T
        data_noise = np.transpose(data_noise, (1, 0, 2))

        delta_kx = self.delta_kx2
        delta_ky = self.delta_ky2

        nb_clusters = len(clusters)
        nb_samples_per_shot, nb_shots = data_mri.shape
        nb_channels = data_noise.shape[2]
        convolution_size = nb_channels * (2 * delta_kx + 1) * (2 * delta_ky + 1)

        # classic: the corrected signals using the classic EDITER algo from the paper,
        # i.e correction of clusters without the optimized coefficients alpha_i for each shot i.
        classic_corrected = np.zeros((nb_samples_per_shot, nb_shots), dtype=TYPE_COMPLEX)

        # Final H matrix for classic EDITER. Contains the computed h_j vector for each cluster j.
        classic_H_mat = np.zeros((convolution_size, nb_clusters), dtype=TYPE_COMPLEX)

        if optimized:
            # optimized: the corrected signals with correction of clusters based on vector h_j common
            # to cluster j + optimization of coefficients alpha_i for each shot i of the cluster,
            # so that the actual impulse response vector for each shot i is alpha_i * h_j.
            optim_corrected = np.zeros((nb_samples_per_shot, nb_shots), dtype=TYPE_COMPLEX)

            # Final H matrix for optimized EDITER. Contains the computed common h_j of each cluster j.
            optim_H_mat = np.zeros(
                (convolution_size, nb_clusters),
                dtype=TYPE_COMPLEX,
            )
            # Final H * alpha matrix for optimized EDITER. Contains for each shot i in cluster j: alpha_i * h_j
            alpha_H_mat = np.zeros(
                (convolution_size, nb_shots),
                dtype=TYPE_COMPLEX,
            )

            # Initialize memory for all alphas
            initial_alphas = (
                np.ones(nb_shots, dtype=TYPE_COMPLEX)
                if initial_alphas is None
                else np.copy(initial_alphas)
            )
            final_alphas = np.zeros(nb_shots, dtype=TYPE_COMPLEX)

        for cluster_idx in range(nb_clusters):
            # For each window i.e. each cluster of shots
            shot_indices = clusters[cluster_idx]
            l = len(shot_indices)

            noise_mat = _construct_noise_mat(data_noise[:, shot_indices], delta_kx, delta_ky)
            E_toeplitz_mat = noise_mat.reshape((convolution_size, -1)).T

            # Reshape signal matrix for this cluster
            sub_signal_mat = data_mri[:, shot_indices].reshape((nb_samples_per_shot, l))

            # Compute classic EDITER h vector for this cluster
            h, _, _, _ = np.linalg.lstsq(
                E_toeplitz_mat, sub_signal_mat.reshape((nb_samples_per_shot * l)), rcond=None
            )
            # Compute classic EDITER Eh
            Eh = np.matmul(E_toeplitz_mat, h).reshape((nb_samples_per_shot, l))

            # Store h in H matrix
            classic_H_mat[:, cluster_idx] = h.flatten()

            # Store classic EDITER corrected shots for this cluster
            classic_corrected[:, shot_indices] = sub_signal_mat - Eh

            if optimized:
                # Coordinate descent to iteratively optimize alphas and h
                alphas = initial_alphas[shot_indices]

                for _ in range(max_iter):
                    # Alpha optimization with fixed h
                    old_alphas = np.copy(alphas)  # keep old alphas to check convergence
                    for shot in range(l):
                        eh_shot = Eh[:, shot].reshape((-1, 1))
                        # find optimal alpha_i (of iteration i) for the shot given h.
                        alpha, _, _, _ = np.linalg.lstsq(
                            eh_shot, sub_signal_mat[:, shot], rcond=None
                        )
                        # The accumulated alpha for the shot is the product of iterative alpha_i.
                        alphas[shot] *= alpha[0]
                        # Update noise matrix to include newly computed optimal alpha_i
                        noise_mat[:, :, shot] *= alpha[0]

                    # h optimization with fixed alphas:
                    old_h = np.copy(h)  # keep old h to check convergence

                    # Compute new block-Toeplitz matrix from updated noise_mat
                    E_toeplitz_mat = noise_mat.reshape((convolution_size, -1)).T

                    # Find optimal h for this cluster given the alphas of each shot from the cluster.
                    h, _, _, _ = np.linalg.lstsq(
                        E_toeplitz_mat, sub_signal_mat.flatten(), rcond=None
                    )
                    Eh = np.matmul(E_toeplitz_mat, h).reshape((nb_samples_per_shot, l))

                    # Check convergence
                    s = np.linalg.norm(h - old_h) + np.linalg.norm(alphas - old_alphas)
                    if s < conv_threshold:
                        break

                # Store results
                final_alphas[shot_indices] = alphas
                optim_H_mat[:, cluster_idx] = h.flatten()
                alpha_H_mat[:, shot_indices] = np.matmul(
                    alphas.reshape((-1, 1)), h.reshape((1, -1))
                ).T

                # Store corrections
                optim_corrected[:, shot_indices] = sub_signal_mat - Eh

        if optimized:
            # FIXME: temporary shape fix until the shape standard is applied in the code
            optim_corrected = optim_corrected.T
            return optim_corrected, optim_H_mat, alpha_H_mat, final_alphas, self.original_H_mat
        else:
            # FIXME: temporary shape fix until the shape standard is applied in the code
            classic_corrected = classic_corrected.T
            return classic_corrected, classic_H_mat


def _construct_noise_mat(data_noise, delta_kx, delta_ky):
    """Constructs a noise matrix to use for the EDITER optimization.

    Args:
        data_noise (array) : shape (nb_samples_per_shot, nb_shots, nb_channels) noise data used
            to construct the matrix.
        delta_kx (int) : convolution size in the "nb_samples_per_shot" direction, i.e within one shot.
        delta_ky (int) : convolution size in the "nb_shots" direction, i.e between shots.

    Returns:
        noise_mat (array) : shape ((2 * delta_kx + 1) * (2 * delta_ky + 1) * nb_channels,
            nb_samples_per_shot, nb_shots), noise matrix.
    """
    data_noise_padded = []
    noise_mat = []  # E for this window
    # Indices of the shots included in this window
    nb_samples_per_shot, nb_shots, nb_channels = data_noise.shape

    for coil_idx in range(nb_channels):
        # Zero pad the noise data to handle extremeties (to apply convolutions "properly").
        data_noise_padded.append(
            np.pad(
                data_noise[:, :, coil_idx].reshape((nb_samples_per_shot, nb_shots)),
                ((delta_kx, delta_kx), (delta_ky, delta_ky)),
                "constant",
            )
        )

    for x_shift in range(-delta_kx, delta_kx + 1):
        for y_shift in range(-delta_ky, delta_ky + 1):
            for coil_idx in range(nb_channels):
                # Shift the noise data by x_shift and y_shift and add it to the noise matrix
                # for each noise channel.
                data_noise_rotated = np.roll(
                    data_noise_padded[coil_idx], (x_shift, y_shift), (0, 1)
                )
                # Size of the padded data : size_x = nb_samples_per_shot + 2 * delta_kx,
                #                           size_y = nb_shots + 2 * delta_ky
                size_x, size_y = data_noise_rotated.shape
                noise_mat.append(
                    data_noise_rotated[delta_kx : size_x - delta_kx, delta_ky : size_y - delta_ky]
                )
    # The final matrix is of size (nb_channels * (2 * delta_kx + 1) * (2 * delta_ky + 1),
    # nb_samples_per_shot, nb_shots)
    noise_mat = np.array(noise_mat)
    return noise_mat

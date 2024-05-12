"""Module containing the augmentation class for random erosion and dilation."""
import numpy as np
import scipy.ndimage as ndimage


class RandomErosion:
    """Random erosion augmentation class."""

    def __init__(
        self, randomState: np.random.RandomState, alpha=0.66, beta=5
    ) -> None:
        """Initialize the random erosion augmentation class.

        Randomly erodes/dilates the image with a probability of alpha and a maximum number of iterations of beta.

        Args:
            randomState (np.random.RandomState): randomstate object to use for random number generation
            alpha (float, optional): Hyperparameter alpha, probability of doing augmentation. Defaults to 0.66.
            beta (int, optional): Hyperparameter beta, maximum number of erosion/dilation iterations. Defaults to 5.
        """
        self.alpha = alpha
        self.beta = beta
        self.randomState = randomState

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        """Apply the augmentation to the image.

        Args:
            img_np (np.ndarray): image to augment

        Returns:
            np.ndarray: augmented image
        """
        img_np = np.where(img_np != 0, 1, 0).astype(img_np.dtype)

        for i in range(img_np.shape[1]):
            do_augment = self.randomState.rand() < self.alpha

            if do_augment:
                do_erosion = self.randomState.rand() < 0.5

                if do_erosion:
                    n_iter = self.randomState.randint(
                        1, self.beta
                    )  # [1, beta)
                    img_np[:, i, :] = ndimage.binary_erosion(
                        img_np[:, i, :], iterations=n_iter
                    ).astype(img_np.dtype)
                else:
                    n_iter = self.randomState.randint(
                        1, self.beta
                    )  # [1, beta)
                    img_np[:, i, :] = ndimage.binary_dilation(
                        img_np[:, i, :], iterations=n_iter
                    ).astype(img_np.dtype)

        return img_np
"""
2D Gaussian Splatting Analysis
https://en.wikipedia.org/wiki/Gaussian_splatting
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class ImageGaussianSplatting:
    def __init__(self, image: np.array):
        """
        Perform Gaussian splatting analysis on a 2D binary image.

        Args:
            image (np.array): 2D binary. The object of interest should be represented by 1s.
        """
        self.image = image

        # Find pixel indices of interest
        self.y, self.x = np.nonzero(image)

        if len(self.x) == 0 or len(self.y) == 0:
            # No pixels of interest
            self.x_c = self.y_c = float('nan')
            self.primary_dir = self.secondary_dir = np.array([0, 0])
            self.angle = self.spread_primary = self.spread_secondary = float('nan')
            return

        if len(self.x) == 1 and len(self.y) == 1:
            # Exactly one pixel of interest
            self.x_c, self.y_c = self.x[0], self.y[0]
            self.primary_dir = self.secondary_dir = np.array([0, 0])
            self.angle = self.spread_primary = self.spread_secondary = 0
            return
        
        # Compute the center (centroid)
        self.x_c = np.mean(self.x)
        self.y_c = np.mean(self.y)

        # Center coordinates
        self.x_centered = self.x - self.x_c
        self.y_centered = self.y - self.y_c

        # Covariance matrix
        cov_matrix = np.cov(np.stack((self.x_centered, self.y_centered)))
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Extract angle and spread
        self.primary_dir = eigenvectors[:, 1] # Eigenvector of the largest eigenvalue
        self.secondary_dir = eigenvectors[:, 0] # Orthogonal eigenvector
        self.angle = np.arctan2(self.primary_dir[1], self.primary_dir[0]) % np.pi  # Angle in [0, π[
        self.spread_primary = np.sqrt(eigenvalues[1])
        self.spread_secondary = np.sqrt(eigenvalues[0])

    def print_results(self):
        print(f"Center: ({self.x_c:.2f}, {self.y_c:.2f})")
        print(f"Angle (radians): {self.angle:.2f}")
        print(f"Primary spread: {self.spread_primary:.2f}")
        print(f"Secondary spread: {self.spread_secondary:.2f}")

    def show(self):
        plt.imshow(self.image, cmap='gray', origin='lower')
        plt.scatter([self.x_c], [self.y_c], color='red', label='Center')

        if not np.isnan(self.angle):
            # Add ellipse to the visualization
            ellipse = Ellipse(
                (self.x_c, self.y_c),
                width=2 * self.spread_primary if self.spread_primary > 0 else 1,
                height=2 * self.spread_secondary if self.spread_secondary > 0 else 1,
                angle=np.degrees(self.angle),
                edgecolor='blue',
                facecolor='none',
                lw=2,
                label='Gaussian Ellipse'
            )
            plt.gca().add_patch(ellipse)

            # Draw primary and secondary directions
            plt.quiver(
                self.x_c, self.y_c, self.primary_dir[0], self.primary_dir[1],
                color='blue', scale=3, label='Primary direction'
            )
            plt.quiver(
                self.x_c, self.y_c, self.secondary_dir[0], self.secondary_dir[1],
                color='green', scale=3, label='Secondary direction'
            )

        # Reversing the y-axis does not work, the quiver's are not rotated correctly
        # So this particular plot is upside down
        # plt.gca().invert_yaxis()
        # plt.xlim(-0.5, self.image.shape[1] - 0.5)
        # plt.ylim(-0.5, self.image.shape[0] - 0.5)

        plt.legend()
        plt.title("2D Gaussian Splatting Analysis")
        plt.show()

if __name__ == "__main__":
    # Generate a test image
    image = np.zeros((100, 100))
    image[30:35, 10:25] = 1
    image[65:75, 90:95] = 1
    image[40:60, 30:70] = 1
    image[60:80, 60:80] = 1

    # Perform Gaussian splatting analysis
    igs = ImageGaussianSplatting(image)
    igs.print_results()
    igs.show()

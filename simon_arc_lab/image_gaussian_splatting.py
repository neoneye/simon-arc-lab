import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class ImageGaussianSplatting:
    def __init__(self, image: np.array):
        self.image = image

        # Find pixel indices of interest
        self.y, self.x = np.nonzero(image)

        # Compute the center (centroid)
        self.x_c = np.mean(self.x)
        self.y_c = np.mean(self.y)

        # Center coordinates
        self.x_centered = self.x - self.x_c
        self.y_centered = self.y - self.y_c

        # Covariance matrix
        cov_matrix = np.cov(np.stack((self.x_centered, self.y_centered)))
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov_matrix)

        # Extract angle and spread
        self.primary_dir = self.eigenvectors[:, 1] # Eigenvector of the largest eigenvalue
        self.secondary_dir = self.eigenvectors[:, 0] # Orthogonal eigenvector
        self.angle = np.arctan2(self.primary_dir[1], self.primary_dir[0]) # Angle in radians
        self.spread_primary = np.sqrt(self.eigenvalues[1])
        self.spread_secondary = np.sqrt(self.eigenvalues[0])

    def print_results(self):
        print(f"Center: ({self.x_c:.2f}, {self.y_c:.2f})")
        print(f"Angle (radians): {self.angle:.2f}")
        print(f"Primary spread: {self.spread_primary:.2f}")
        print(f"Secondary spread: {self.spread_secondary:.2f}")

    def show(self):
        # Visualize the results
        plt.imshow(self.image, cmap='gray', origin='lower')
        plt.scatter([self.x_c], [self.y_c], color='red', label='Center')
        
        # Add ellipse to the visualization
        ellipse = Ellipse(
            (self.x_c, self.y_c),
            width=2 * self.spread_primary,
            height=2 * self.spread_secondary,
            angle=np.degrees(self.angle),
            edgecolor='blue',
            facecolor='none',
            lw=2,
            label='Gaussian Ellipse'
        )
        plt.gca().add_patch(ellipse)

        plt.quiver(
            self.x_c, self.y_c, self.primary_dir[0], self.primary_dir[1],
            color='blue', scale=3, label='Primary direction'
        )
        plt.quiver(
            self.x_c, self.y_c, self.secondary_dir[0], self.secondary_dir[1],
            color='green', scale=3, label='Secondary direction'
        )
        plt.legend()
        plt.title("2D Gaussian Splatting Analysis")
        plt.show()

if __name__ == "__main__":
    # Generate a test image
    image = np.zeros((100, 100))
    image[90:95, 70:75] = 1
    image[30:70, 30:70] = 1
    image[60:80, 60:80] = 1

    # Perform Gaussian splatting analysis
    igs = ImageGaussianSplatting(image)
    igs.print_results()
    igs.show()

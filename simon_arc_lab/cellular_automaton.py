# Variants of cellular automata
# IDEA: wire world
import numpy as np

class CARule:
    def __init__(self):
        pass

    def apply(self, image):
        new_image = image.copy()
        height, width = image.shape
        for i in range(height):
            for j in range(width):
                # Count the number of alive neighbors
                alive_neighbors = (
                    image[i, (j-1)%width] + image[i, (j+1)%width] +
                    image[(i-1)%height, j] + image[(i+1)%height, j] +
                    image[(i-1)%height, (j-1)%width] + image[(i-1)%height, (j+1)%width] +
                    image[(i+1)%height, (j-1)%width] + image[(i+1)%height, (j+1)%width]
                )
                # Apply the rules of the cellular automaton
                new_image[i, j] = self.rule(image[i, j], alive_neighbors)
        return new_image
    
    def rule(self, center: int, alive_count: int) -> int:
        raise NotImplementedError()

class CARuleGameOfLife(CARule):
    def rule(self, center: int, alive_count: int) -> int:
        """
        Apply one step of the Game of Life to the given image.
        https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
        """
        if center == 1:
            if alive_count < 2 or alive_count > 3:
                return 0
        else:
            if alive_count == 3:
                return 1
        return center

class CARuleHighLife(CARule):
    def rule(self, center: int, alive_count: int) -> int:
        """
        Apply one step of the HighLife to the given image.
        https://en.wikipedia.org/wiki/Highlife_%28cellular_automaton%29
        https://conwaylife.com/wiki/OCA:HighLife
        """
        if center == 1:
            if alive_count < 2 or alive_count > 3:
                return 0
        else:
            if alive_count == 3 or alive_count == 6:
                return 1
        return center

class CARuleServiettes(CARule):
    def rule(self, center: int, alive_count: int) -> int:
        """
        Apply one step of the Serviettes to the given image.
        https://conwaylife.com/wiki/OCA:Serviettes
        """
        if center == 0:
            if alive_count == 2 or alive_count == 3 or alive_count == 4:
                return 1
        return 0

def cellular_automata_gameoflife_wrap(image):
    return CARuleGameOfLife().apply(image)

def cellular_automata_highlife_wrap(image):
    return CARuleHighLife().apply(image)

def cellular_automata_serviettes_wrap(image):
    return CARuleServiettes().apply(image)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Define the size of the grid
    GRID_SIZE = 100

    # Initialize the grid with random values (0 or 1)
    grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE))
    # grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    # grid[GRID_SIZE//2, GRID_SIZE//2] = 1
    # grid[GRID_SIZE//2, GRID_SIZE//2 + 1] = 1
    # grid[GRID_SIZE//2 + 1, GRID_SIZE//2] = 1
    # grid[GRID_SIZE//2 + 1, GRID_SIZE//2 + 1] = 1

    # Set up the figure and axis
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='binary')

    def animate(frame):
        global grid
        grid = cellular_automata_gameoflife_wrap(grid)
        # grid = cellular_automata_highlife_wrap(grid)
        # grid = cellular_automata_serviettes_wrap(grid)
        img.set_data(grid)
        return [img]

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=200, interval=100, blit=True)

    # Display the animation
    plt.show()

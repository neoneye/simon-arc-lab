# Variants of cellular automata
# IDEA: high life
# IDEA: serviettes life (persian rug)
# IDEA: wire world
import numpy as np

def cellular_automata_gameoflife_wrap(image):
    """
    Apply one step of the Game of Life to the given image.
    https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
    """
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
            # Apply the rules of the Game of Life
            if image[i, j] == 1:
                if alive_neighbors < 2 or alive_neighbors > 3:
                    new_image[i, j] = 0
            else:
                if alive_neighbors == 3:
                    new_image[i, j] = 1
    return new_image

def cellular_automata_highlife_wrap(image):
    """
    Apply one step of the HighLife to the given image.
    https://en.wikipedia.org/wiki/Highlife_%28cellular_automaton%29
    https://conwaylife.com/wiki/OCA:HighLife
    """
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
            # Apply the rules of the HighLife
            if image[i, j] == 1:
                if alive_neighbors < 2 or alive_neighbors > 3:
                    new_image[i, j] = 0
            else:
                if alive_neighbors == 3 or alive_neighbors == 6:
                    new_image[i, j] = 1
    return new_image

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Define the size of the grid
    GRID_SIZE = 100

    # Initialize the grid with random values (0 or 1)
    grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE))

    # Set up the figure and axis
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='binary')

    def animate(frame):
        global grid
        # grid = cellular_automata_gameoflife_wrap(grid)
        grid = cellular_automata_highlife_wrap(grid)
        img.set_data(grid)
        return [img]

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=200, interval=100, blit=True)

    # Display the animation
    plt.show()

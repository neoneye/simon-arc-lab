# Variants of cellular automata
# IDEA: high life
# IDEA: serviettes life (persian rug)
# IDEA: wire world
import numpy as np

def game_of_life_wrap(grid):
    new_grid = grid.copy()
    height, width = grid.shape
    for i in range(height):
        for j in range(width):
            # Count the number of alive neighbors
            alive_neighbors = (
                grid[i, (j-1)%width] + grid[i, (j+1)%width] +
                grid[(i-1)%height, j] + grid[(i+1)%height, j] +
                grid[(i-1)%height, (j-1)%width] + grid[(i-1)%height, (j+1)%width] +
                grid[(i+1)%height, (j-1)%width] + grid[(i+1)%height, (j+1)%width]
            )
            # Apply the rules of the Game of Life
            if grid[i, j] == 1:
                if alive_neighbors < 2 or alive_neighbors > 3:
                    new_grid[i, j] = 0
            else:
                if alive_neighbors == 3:
                    new_grid[i, j] = 1
    return new_grid

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
        grid = game_of_life_wrap(grid)
        img.set_data(grid)
        return [img]

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=200, interval=100, blit=True)

    # Display the animation
    plt.show()

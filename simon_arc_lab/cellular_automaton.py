# Variants of cellular automata
import numpy as np

class CARule:
    def apply(self, image):
        new_image = image.copy()
        height, width = image.shape
        for y in range(height):
            for x in range(width):
                # Traverse all 8 neighbors and determine if they are alive
                count_exactly_one = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        value = image[(y+dy)%height, (x+dx)%width]
                        if value == 1:
                            count_exactly_one += 1
                # Apply the rules of the cellular automaton
                new_image[y, x] = self.rule(image[y, x], count_exactly_one)
        return new_image
    
    def rule(self, center: int, alive_count: int) -> int:
        raise NotImplementedError()

class CARuleGameOfLife(CARule):
    def rule(self, center: int, alive_count: int) -> int:
        """
        Apply one step of the Game of Life.
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
        Apply one step of the HighLife.
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
        Apply one step of the Serviettes.
        https://conwaylife.com/wiki/OCA:Serviettes
        """
        if center == 0:
            if alive_count == 2 or alive_count == 3 or alive_count == 4:
                return 1
        return 0

class CARuleWireWorld(CARule):
    def rule(self, center: int, alive_count: int) -> int:
        """
        https://en.wikipedia.org/wiki/Wireworld

        0 = empty
        1 = electron head
        2 = electron tail
        3 = conductor
        """
        if center == 1: # electron head
            return 2 # electron tail

        if center == 2: # electron tail
            return 3 # conductor
        
        if center == 3 and (alive_count == 1 or alive_count == 2): # conductor with 1 or 2 electron heads
            return 1 # electron head

        return center

class CARuleCave(CARule):
    def rule(self, center: int, alive_count: int) -> int:
        """
        Create a cave system
        https://www.roguebasin.com/index.php/Cellular_Automata_Method_for_Generating_Random_Cave-Like_Levels
        http://pixelenvy.ca/wa/ca_cave.html
        """
        if alive_count < 4:
            return 0
        if alive_count > 5:
            return 1
        return center

class CARuleMaze(CARule):
    def rule(self, center: int, alive_count: int) -> int:
        """
        Create a maze
        https://conwaylife.com/wiki/OCA:Maze
        """
        if alive_count == 3:
            return 1
        if alive_count < 1 or alive_count > 5:
            return 0
        return center

def cellular_automata_gameoflife_wrap(image):
    return CARuleGameOfLife().apply(image)

def cellular_automata_highlife_wrap(image):
    return CARuleHighLife().apply(image)

def cellular_automata_serviettes_wrap(image):
    return CARuleServiettes().apply(image)

def cellular_automata_wireworld_wrap(image):
    return CARuleWireWorld().apply(image)

def cellular_automata_cave_wrap(image):
    return CARuleCave().apply(image)

def cellular_automata_maze_wrap(image):
    return CARuleMaze().apply(image)

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
        # grid = cellular_automata_cave_wrap(grid)
        # grid = cellular_automata_maze_wrap(grid)
        img.set_data(grid)
        return [img]

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=200, interval=100, blit=True)

    # Display the animation
    plt.show()

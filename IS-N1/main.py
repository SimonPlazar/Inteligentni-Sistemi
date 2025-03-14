import numpy as np
import matplotlib.pyplot as plt


class CellularAutomaton1D:
    def __init__(self, rule, width, generations):
        """
        Initialize 1D cellular automaton
        :param rule: Rule number (0-255)
        :param width: Width of the grid
        :param generations: Number of generations to simulate
        """
        self.rule = rule
        self.width = width
        self.generations = generations
        self.grid = np.zeros((generations, width), dtype=int)
        self.rule_binary = self._get_rule_binary(rule)

        # Initialize the first generation with a single cell in the middle
        self.grid[0, width // 2] = 1

    def _get_rule_binary(self, rule):
        """Convert rule number to binary representation (8 bits)"""
        return [(rule >> i) & 1 for i in range(8)]

    def _get_new_state(self, left, center, right):
        """Determine new state based on neighborhood and rule"""
        # Calculate the index into the rule binary (0-7)
        index = 4 * left + 2 * center + right
        return self.rule_binary[7 - index]

    def generate(self):
        """Generate all generations"""
        for i in range(1, self.generations):
            for j in range(self.width):
                # Get neighboring cells with wrap-around
                left = self.grid[i - 1, (j - 1) % self.width]
                center = self.grid[i - 1, j]
                right = self.grid[i - 1, (j + 1) % self.width]

                # Apply rule
                self.grid[i, j] = self._get_new_state(left, center, right)
        return self.grid

    def display(self):
        """Display the automaton using matplotlib"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary')
        plt.title(f"Elementary Cellular Automaton - Rule {self.rule}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    print("1D Cellular Automaton - Elementary Rules")

    while True:
        try:
            rule = int(input("Enter rule number (0-255): "))
            if 0 <= rule <= 255:
                break
            else:
                print("Rule must be between 0 and 255.")
        except ValueError:
            print("Please enter a valid integer.")

    width = 101  # Width of the grid (odd number for centered pattern)
    generations = 50  # Number of generations

    ca = CellularAutomaton1D(rule, width, generations)
    ca.generate()
    ca.display()


if __name__ == "__main__":
    main()
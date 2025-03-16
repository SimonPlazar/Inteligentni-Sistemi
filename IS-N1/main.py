import numpy as np
import matplotlib.pyplot as plt


def getBinary(rule):
    # return [int(bit) for bit in f"{rule:08b}"]
    return [(rule >> i) & 1 for i in range(8)]


class CellularAutomaton1D:
    def __init__(self, rule, width, generations):
        self.rule = rule
        self.width = width
        self.generations = generations
        self.grid = np.zeros((generations, width), dtype=int)
        self.rule_binary = getBinary(rule)

        self.grid[0, width // 2] = 1  # first generation with a single cell in the middle

    def getState(self, left, center, right):
        # Calculate the index into the rule binary (0-7)
        index = 4 * left + 2 * center + right
        return self.rule_binary[index]
        # return self.rule_binary[7 - index]

    def generate(self):
        for i in range(1, self.generations):
            for j in range(self.width):
                # Get neighboring cells with wrap-around
                left = self.grid[i - 1, (j - 1) % self.width]
                center = self.grid[i - 1, j]
                right = self.grid[i - 1, (j + 1) % self.width]

                # Apply rule
                self.grid[i, j] = self.getState(left, center, right)
        return self.grid

    def display(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary')
        plt.title(f"Elementary Cellular Automaton - Rule {self.rule}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def CellularAutomaton_1D():
    print("1D Elementary Cellular Automaton")

    while True:
        try:
            rule = int(input("Enter rule number (0-255): "))
            if 0 <= rule <= 255:
                break
            else:
                print("Rule must be between 0 and 255.")
        except ValueError:
            print("Please enter a valid integer.")

    width = 101
    generations = 70

    ca = CellularAutomaton1D(rule, width, generations)
    ca.generate()
    ca.display()


if __name__ == "__main__":
    CellularAutomaton_1D()

from Automaton_1D import CellularAutomaton_1D
from Automaton_2D import CellularAutomaton_2D


if __name__ == "__main__":

    # CellularAutomaton_2D()

    mode = input("Select mode (1 for 1D, 2 for 2D): ")
    if mode == "1":
        CellularAutomaton_1D()
    elif mode == "2":
        CellularAutomaton_2D()
    else:
        print("Invalid mode. Defaulting to 1D automaton.")
        CellularAutomaton_1D()

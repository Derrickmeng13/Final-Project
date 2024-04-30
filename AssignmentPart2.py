import numpy as np
import matplotlib.pyplot as plt
import argparse

# Define the Deffuant model
def defuant_main(N, T, beta, steps=1000):
    opinions = np.random.rand(N)
    opinions_history = []

    for step in range(steps):
        i, j = np.random.choice(N, 2, replace=False)
        if abs(opinions[i] - opinions[j]) < T:
            opinions[i] += beta * (opinions[j] - opinions[i])
            opinions[j] += beta * (opinions[i] - opinions[j])

        opinions_history.append(opinions.copy())

    # plot two charts
    def plot_opinions(opinions_history, N, beta, T):
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.hist(opinions, bins=20, color='blue', alpha=0.7)
        plt.title('Opinion Distribution at Final Timestep')
        plt.xlabel('Opinion')
        plt.ylabel('Frequency')
        plt.title(f"Final Opinion Distribution (Coupling:{beta},Threshold:{T})")

        plt.subplot(1, 2, 2)
        for i in range(N):
            plt.plot(range(len(opinions_history)), [h[i] for h in opinions_history], color='red', linewidth=0.5)
        plt.title('Individual Opinions Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Opinion')
        plt.title(f"Opinion Evolution (Coupling:{beta},Threshold:{T})")

        plt.tight_layout()
        plt.show()

    plot_opinions(opinions_history, N, beta, T)
    return opinions_history

# Run simulation
def run_defuant_main(T, beta):
    defuant_main(N=100, T=T, beta=beta)

# test function
def test_defuant_main():
    run_defuant_main(T=0.5, beta=0.5)
    run_defuant_main(T=0.1, beta=0.5)
    run_defuant_main(T=0.5, beta=0.1)
    run_defuant_main(T=0.1, beta=0.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defuant Model")
    parser.add_argument("-defuant", action="store_true", help="Run Defuant model")
    parser.add_argument("-test_defuant", action="store_true", help="Run tests for Defuant model")
    parser.add_argument("-threshold", type=float, default=0.2, help="Interaction threshold")
    parser.add_argument("-beta", type=float, default=0.2, help="Coupling parameter")
    args = parser.parse_args()

    if args.defuant:
        run_defuant_main(args.threshold, args.beta)
    elif args.test_defuant:
        test_defuant_main()

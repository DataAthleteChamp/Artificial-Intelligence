import numpy as np
import matplotlib.pyplot as plt
import math


def binary_to_real(binary, min_val=-8, max_val=8):
    return min_val + (int(binary, 2) / (2 ** len(binary) - 1)) * (max_val - min_val)


def real_to_binary(value, length=16):
    value = int((value - (-8)) / (8 - (-8)) * (2 ** length - 1))
    return format(value, f'0{length}b')


# wartosc f osobnika
def evaluate(individual, func):
    real_val = binary_to_real(individual)
    return func(real_val)


# selekcja turniejowa k-liczba uczestników turnieju
def select(population, scores, k=3):
    selected = []
    for _ in range(len(population)):
        indices = np.random.randint(0, len(population), k)
        best_idx = indices[np.argmin([scores[i] for i in indices])]
        selected.append(population[best_idx])
    return selected


# krzyżowanie jednopunktowe między dwoma rodzicami
# tworzy nowe chromosomy poprzez wymieszanie części genotypów dwóch rodziców.
def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2


# mutacje do chromosomu osobnika
# zmienia losowo wybrane bity w chromosomie
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual = individual[:i] + str(1 - int(individual[i])) + individual[i + 1:]
    return individual


def validate_function(user_function):
    try:
        test_x = np.random.uniform(-8, 8)
        eval(user_function, {"x": test_x, "np": np, "math": math})
        return True
    except:
        return False


print("""
Wprowadź funkcję jednoargumentową używając 'x' jako zmiennej

Przykłady:
- Funkcja kwadratowa: 'x**2 - 5*x + 4'
- Funkcja trygonometryczna: 'np.sin(x)'
- Funkcja eksponencjalna: 'np.exp(x)'
- Funkcja logarytmiczna: 'np.log(x)'
- Funkcja trzeciego stopnia: 'x**3 - 3*x**2 + 2*x - 5'
- Funkcja czwartego stopnia: 'x**4 - 10*x**3 + 35*x**2 - 50*x + 24'
- Funkcja piątego stopnia: '2*x**5 - 5*x**4 + x**3 + 8*x**2 - 20*x + 16'

""")

user_function = input("Podaj funkcję: ")
while not validate_function(user_function):
    print("Błąd w funkcji, popraw składnię")
    user_function = input("Podaj funkcję ponownie: ")


# func = lambda x: eval(user_function, {"x": x, "np": np, "math": math})
def safe_log(x):
    if x <= 0:
        return float('inf')
    else:
        return np.log(x)


func = lambda x: safe_log(x) if user_function == 'np.log(x)' else eval(user_function, {"x": x, "np": np, "math": math})


# algorytm genetyczny
def genetic_algorithm(func, population_size, generations, mutation_rate, crossover_rate, no_change_limit,
                      improvement_threshold):
    population = [real_to_binary(np.random.uniform(-8, 8)) for _ in range(population_size)]
    scores = [evaluate(ind, func) for ind in population]

    best_scores = []
    best_score = min(scores)
    no_change_count = 0

    for generation in range(generations):
        selected = select(population, scores)
        population = []
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        scores = [evaluate(ind, func) for ind in population]

        current_best_score = min(scores)
        best_scores.append(current_best_score)

        # sprawdzanie czy obecny wynik jest leoszy od najleoszego
        if current_best_score <= best_score * (1 - improvement_threshold):
            best_score = current_best_score
            no_change_count += 1
        else:
            no_change_count = 0

        # zatrzymannie alg jesli nie było widocznej poprawy przez ileś iteracji
        if no_change_count >= no_change_limit:
            print(f"Algorytm zatrzymany po {generation + 1} generacjach. Brak znaczącej poprawy alg")
            break

    return best_scores, generation + 1


# parametry algorytmu
population_size = 200
generations = 100
mutation_rate = 0.08
crossover_rate = 0.9
no_change_limit = 10
improvement_threshold = 0.01  # 1e-6

best_scores, last_generation = genetic_algorithm(
    func,
    population_size=population_size,
    generations=generations,
    mutation_rate=mutation_rate,
    crossover_rate=crossover_rate,
    no_change_limit=no_change_limit,
    improvement_threshold=improvement_threshold)

plt.plot(best_scores)
plt.xlabel('Generacja')
plt.ylabel('Najlepszy wynik')
plt.title('Postęp Algorytmu Genetycznego')
# plt.savefig('step2.png')
plt.show()

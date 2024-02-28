import random  # Импорт модуля для генерации случайных чисел
import numpy as np  # Импорт модуля numpy для работы с массивами
import networkx as nx  # Импорт библиотеки NetworkX для работы с графами
import matplotlib.pyplot as plt  # Импорт модуля для отрисовки графиков

# Создание случайного графа
def generate_random_graph(num_nodes, seed=None):
    G = nx.complete_graph(num_nodes)  # Создание полного графа с заданным количеством узлов
    random.seed()  # Установка seed для повторяемости результатов
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 100)  # Установка случайных весов рёбер
        print(G.edges[u, v]['weight'])
    return G

# Расчет длины маршрута
def calculate_path_length(path, graph):
    length = 0
    for i in range(len(path)):
        length += graph[path[i]][path[(i + 1) % len(path)]]['weight']  # Суммирование весов рёбер
        print(length)
    return length

# Инициализация начальной популяции
def initialize_population(pop_size, num_nodes):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_nodes))  # Создание случайной последовательности узлов графа
        random.shuffle(individual)  # Перемешивание узлов
        population.append(individual)
    return population

# Оценка приспособленности особи
def evaluate_population(population, graph):
    fitness_scores = []
    for individual in population:
        fitness = 1 / calculate_path_length(individual, graph)  # Оценка приспособленности: чем короче путь, тем лучше
        fitness_scores.append(fitness)
    return fitness_scores

# Селекция родителей для скрещивания (турнирный отбор)
def select_parents(population, fitness_scores, tournament_size):
    parents = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)  # Выбор случайных особей для турнира
        tournament_scores = [fitness_scores[i] for i in tournament_indices]  # Получение оценок приспособленности для участников турнира
        winner_index = tournament_indices[np.argmax(tournament_scores)]  # Выбор победителя турнира
        parents.append(population[winner_index])  # Добавление победителя в список родителей
    return parents

# Одноточечное скрещивание
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)  # Выбор случайной точки для скрещивания
    child = parent1[:crossover_point]  # Создание первой части потомка от первого родителя
    for gene in parent2:  # Перебор генов второго родителя
        if gene not in child:
            child.append(gene)  # Добавление гена в потомка, если его там нет
    return child

# Мутация (обмен двух генов)
def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)  # Выбор двух случайных индексов для мутации
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]  # Обмен выбранных генов

# Получение лучшего индивида из популяции
def get_best_individual(population, fitness_scores):
    return population[np.argmax(fitness_scores)]  # Возвращение индивида с наибольшей приспособленностью

# Генетический алгоритм
def genetic_algorithm(graph, pop_size, num_generations, tournament_size):
    population = initialize_population(pop_size, len(graph.nodes()))  # Инициализация начальной популяции
    for _ in range(num_generations):  # Цикл по количеству поколений
        fitness_scores = evaluate_population(population, graph)  # Оценка приспособленности популяции
        parents = select_parents(population, fitness_scores, tournament_size)  # Выбор родителей для скрещивания
        offspring_population = []  # Создание списка для потомков
        for i in range(0, pop_size, 2):  # Цикл по парам родителей
            child1 = crossover(parents[i], parents[i+1])  # Скрещивание пары родителей
            child2 = crossover(parents[i+1], parents[i])  # Скрещивание пары родителей (в обратном порядке)
            offspring_population.extend([child1, child2])  # Добавление потомков в популяцию
        for individual in offspring_population:  # Цикл по потомкам
            if random.random() < mutation_rate:  # Применение мутации с определенной вероятностью
                mutate(individual)
        population = offspring_population  # Обновление популяции на следующее поколение
    best_individual = get_best_individual(population, evaluate_population(population, graph))  # Получение лучшего индивида
    return best_individual

# Параметры алгоритма
num_nodes = 5  # Количество городов
pop_size = 100  # Размер популяции
num_generations = 1000  # Количество поколений
tournament_size = 5  # Размер турнира для селекции
mutation_rate = 0.1  # Вероятность мутации

# Создание графа
graph = generate_random_graph(num_nodes, seed=42)

# Запуск генетического алгоритма
best_path = genetic_algorithm(graph, pop_size, num_generations, tournament_size)

# Вывод результатов
print("Лучший найденный маршрут:", best_path)
print("Длина маршрута:", calculate_path_length(best_path, graph))

# Отрисовка графа
pos = nx.spring_layout(graph)  # Вычисление позиций узлов для отрисовки
nx.draw(graph, pos, with_labels=True, node_color='skyblue')  # Отрисовка графа
nx.draw_networkx_edges(graph, pos, edgelist=[(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)], width=4, edge_color='b')  # Отрисовка лучшего маршрута
nx.draw_networkx_edges(graph, pos, edgelist=[(best_path[-1], best_path[0])])  # Соединение последнего узла с первым для замкнутого маршрута

plt.show()  # Отображение графика

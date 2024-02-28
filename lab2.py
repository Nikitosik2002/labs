import random
import numpy as np


# Алгоритм оптимизации роя частиц (Particle Swarm Optimization, PSO)
class Particle:
    def __init__(self, dim, min_values, max_values):
        # Когда создаётся новая частица, она получает случайные начальные координаты и скорость
        self.position = np.array([random.uniform(min_values[i], max_values[i]) for i in range(dim)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(dim)])
        # При рождении лучшая позиция частицы - это её начальная позиция, так как частица только что создана
        self.best_position = self.position.copy()
        # Изначально приспособленность частицы считается бесконечной, так как мы ещё не оценили, насколько хорошо эта частица подходит для решения задачи
        self.fitness = float('inf')

    def update_velocity(self, global_best_position, omega, phi_p, phi_g):
        # Формула для обновления скорости частицы
        r_p = np.random.rand(len(self.position))
        r_g = np.random.rand(len(self.position))
        # Параметры omega, phi_p и phi_g определяют влияние лучших позиций частиц и глобальной лучшей позиции на обновление скорости
        self.velocity = omega * self.velocity + phi_p * r_p * (self.best_position - self.position) + phi_g * r_g * (global_best_position - self.position)

    def update_position(self, min_values, max_values):
        # Обновляем позицию частицы, учитывая её скорость, и удерживаем её в границах пространства поиска
        self.position = np.clip(self.position + self.velocity, min_values, max_values)

class PSO:
    def __init__(self, num_particles, dim, min_values, max_values, max_iter):
        # Создаём множество частиц
        self.num_particles = num_particles
        self.dim = dim
        self.min_values = min_values
        self.max_values = max_values
        self.max_iter = max_iter
        # Создаём частицы с случайными начальными координатами и скоростями
        self.particles = [Particle(dim, min_values, max_values) for _ in range(num_particles)]
        # Изначально лучшая позиция и приспособленность находятся на бесконечности, так как ещё не оценили ни одну частицу
        self.global_best_position = np.array([random.uniform(min_values[i], max_values[i]) for i in range(dim)])
        self.global_best_fitness = float('inf')

    def optimize(self, objective_function):
        # Основной цикл оптимизации
        for _ in range(self.max_iter):
            # Проходим по всем частицам и оцениваем их приспособленность
            for particle in self.particles:
                particle.fitness = objective_function(particle.position)
                # Если текущая позиция частицы лучше её лучшей позиции, то обновляем лучшую позицию частицы
                if particle.fitness < objective_function(particle.best_position):
                    particle.best_position = particle.position.copy()
                # Если приспособленность частицы лучше глобальной лучшей приспособленности, то обновляем глобальную лучшую позицию
                if particle.fitness < self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = particle.fitness
            # Обновляем скорость и позицию каждой частицы на основе глобальной лучшей позиции
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, 0.5, 0.5, 0.5)
                particle.update_position(self.min_values, self.max_values)
        # Возвращаем найденную глобальную лучшую позицию и приспособленность
        return self.global_best_position, self.global_best_fitness

# Пример функции для оптимизации (можно заменить на свою)
def sphere_function(x):
    # Простая функция сферы: сумма квадратов координат
    return sum([xi**2 for xi in x])

# Создаём экземпляр алгоритма PSO
# PSO (Particle Swarm Optimization) - это метод оптимизации, который моделирует поведение роя частиц в пространстве поиска оптимального решения
# Частицы (потенциальные решения) движутся по пространству поиска, руководствуясь лучшими позициями среди себя и глобальными лучшими позициями во всём рое
# Цель - найти оптимальное решение, минимизируя заданную целевую функцию
num_particles = 30
dim = 3
min_values = [-5, -5, -5]
max_values = [5, 5, 5]
max_iter = 100

# Создаём экземпляр PSO
pso = PSO(num_particles, dim, min_values, max_values, max_iter)

# Запускаем PSO для оптимизации целевой функции (sphere_function)
best_position, best_fitness = pso.optimize(sphere_function)

# Выводим результаты оптимизации
print("Лучшее решение:", best_position)
print("Лучшая приспособленность:", best_fitness)

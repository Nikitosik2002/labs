# Импортируем необходимые модули для работы с MapReduce и генерации случайных чисел
from mrjob.job import MRJob
from mrjob.step import MRStep
import random

# Определяем класс для MapReduce задачи кросс-корреляции для пар товаров
class CrossCorrelationPairs(MRJob):

    # Определяем шаги задачи MapReduce
    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer)
        ]

    # Метод mapper - принимает строку (заказ) и генерирует пары товаров
    def mapper(self, _, line):
        # Разбиваем строку заказа на отдельные товары
        items = line.split()  
        # Перебираем все пары товаров в заказе
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                # Генерируем пару товаров и отправляем в reducer счетчик 1
                yield (items[i], items[j]), 1

    # Метод reducer - суммирует счетчики для каждой пары товаров
    def reducer(self, key, values):
        # Суммируем счетчики для каждой пары товаров
        yield key, sum(values)


# Определяем класс для MapReduce задачи кросс-корреляции по полосам
class CrossCorrelationStripes(MRJob):

    # Определяем шаги задачи MapReduce
    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer)
        ]

    # Метод mapper - для каждого товара в заказе создает полосу с другими товарами и отправляет их в reducer
    def mapper(self, _, line):
        # Разбиваем строку заказа на отдельные товары
        items = line.split()  
        # Перебираем каждый товар в заказе
        for i in range(len(items)):
            stripe = {}
            # Создаем "полосу" для текущего товара
            for j in range(len(items)):
                if i != j:
                    stripe[items[j]] = stripe.get(items[j], 0) + 1
            # Отправляем полосу в reducer
            yield items[i], stripe

    # Метод reducer - собирает и суммирует все полосы для каждого товара
    def reducer(self, key, values):
        total_stripe = {}
        for stripe in values:
            # Суммируем все полосы для текущего товара
            for item, count in stripe.items():
                total_stripe[item] = total_stripe.get(item, 0) + count
        # Возвращаем все полосы для текущего товара
        yield key, total_stripe


# Генерация случайного заказа
def generate_order(items, max_items_per_order):
    num_items = random.randint(1, max_items_per_order)
    return ' '.join(random.choices(items, k=num_items))

# Генерация базы данных заказов
def generate_orders(num_orders, items, max_items_per_order):
    # Генерируем список заказов
    orders = [generate_order(items, max_items_per_order) for _ in range(num_orders)]
    return orders

# Список товаров
items = ['item1', 'item2', 'item3', 'item4', 'item5']

# Генерация заказов
orders = generate_orders(1000, items, 5)  # Генерация 1000 заказов

# Запуск задачи MapReduce по кросс-корреляции по полосам
if __name__ == '__main__':
    CrossCorrelationPairs.run()
    CrossCorrelationStripes.run()

# Вывод примера заказов
for order in orders[:5]:
    print(order)

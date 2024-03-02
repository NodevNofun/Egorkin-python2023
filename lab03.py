import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  


# Начало работы
def main():
    ## 1. Загрузка данных
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    print('Первые десять записей исходных данных')
    print(data[:10])  # Вывести первых десять записей из обучающего набора

    ## 2. Нормализация исходных данных
    x = data[:, 0:2]  # формируем x из двух первых столбцов обучающей выборки
    y = data[:, 2]  # формируем y из третьего столбца обучающей выборки
    m = x.shape[0]  # размер данных
    x_n, mu, sigma = feature_normalize(x)
    X = np.hstack((np.ones((m, 1)), x_n))  # добавление к X фиктивного столбца единиц
    print('Нормализованные данные (10 записей):')
    print(X[:10])
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    ## 3. Градиентный спуск
    print('\nГрадиентный спуск')
    alpha = 0.01  # шаг алгоритма
    iterations = 400  # число итераций
    theta = np.zeros(3)  # начальные theta нулевые
    theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
    print('Найденный градиентным спуском theta:')
    print(theta)
    # Еще несколько вызовов с различными параметрами alpha
    alpha = 1.35
    theta01 = np.zeros(3)
    theta01, J_history01 = gradient_descent(X, y, theta01, alpha, iterations)
    alpha = 0.03
    theta003 = np.zeros(3)
    theta003, J_history003 = gradient_descent(X, y, theta003, alpha, iterations)
    alpha = 0.001
    theta0001 = np.zeros(3)
    theta0001, J_history0001 = gradient_descent(X, y, theta0001, alpha, iterations)
    # Отобразим графики сходимости градиентного спуска
    n = np.linspace(1, iterations, iterations)
    plt.figure()
    plt.xlabel('Номер итерации')
    plt.ylabel('J')
    plt.title('alpha=0.01')
    plt.plot(n, J_history)
    plt.show()
    plt.pause(2)
    plt.figure()
    plt.xlabel('Номер итерации')
    plt.ylabel('J')
    plt.title('alpha=1.35')
    plt.plot(n, J_history01)
    plt.show()
    plt.pause(2)
    plt.figure()
    plt.xlabel('Номер итерации')
    plt.ylabel('J')
    plt.title('alpha=0.03')
    plt.plot(n, J_history003)
    plt.show()
    plt.pause(2)
    plt.figure()
    plt.xlabel('Номер итерации')
    plt.ylabel('J')
    plt.title('alpha=0.001')
    plt.plot(n, J_history0001)
    plt.show()
    plt.pause(2)
    # Зададим построенной модели для предсказания дома площадью 1650 м2 с 3мя этажами
    ex = np.array([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]])
    price = np.dot(ex, theta)
    print('Предсказанная стоимость дома с 3мя этажами и площадью 1650 м2:', price)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    ## 4. Нормальные уравнения
    print('\nНормальные уравнения')
    Xn = np.hstack((np.ones((m, 1)), x))
    theta = normal_equation(Xn, y)
    print('Найденный нормальным уравнением theta:');
    print(theta)
    ex = [1, 1650, 3]
    price = np.dot(ex, theta)
    print('Предсказанная стоимость дома с 3мя этажами и площадью 1650 м2:', price)

    input('Перейдите в терминал и нажмите Enter для завершения')


# Нормализация данных
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


# Функция стоимости
def compute_cost(X, Y, Theta):
    m = X.shape[0]
    J = 1 / (2 * m) * np.sum((np.dot(X, Theta) - Y) ** 2)
    return J


# Алгоритм градиентного спуска
def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    th = theta
    for iter in range(num_iters):
        gradient = np.dot(X.T, (np.dot(X, th) - y)) / m
        th = th - alpha * gradient
        J_history[iter] = compute_cost(X, y, th)
    theta = th
    return theta, J_history


# Получение theta методом нормальных уравнений
def normal_equation(X, y):
    theta = np.linalg.inv(np.dot(X.T, X)) @ np.dot(X.T, y)
    return theta


if __name__ == '__main__':
    plt.ion()
    main()
    plt.clf()

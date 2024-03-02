import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Начало работы
def main():
    ## 1. Загрузка данных
    data = np.loadtxt('ex1data1.txt', delimiter=',')  # чтение данных из текстового файла
    x = data[:, 0]  # значения x в первом столбце файла
    y = data[:, 1]  # значения y во втором столбце файла

    ## 2. Отображение выборки
    plt.scatter(x, y)
    plt.xlabel('Население')
    plt.ylabel('Прибыль')
    plt.show()
    plt.pause(5)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # 3. Проверяем функцию стоимости
    print('\nФункция стоимости')
    m = x.shape[0]  # размер данных
    X = np.hstack((np.ones((m, 1)), x.reshape(-1, 1)))  # преобразуем x в вертикальный вектор и добавим слева фиктивный столбец единиц

    theta = np.zeros(2)  # формируем  вектор из двух нулей
    J = compute_cost(X, y, theta)
    print('Значение theta = [0 ; 0]\Функция стоимости =', J)
    print('Ожидаемое значение функции стоимости (приблизительно): 32.07')
    theta = np.array([-1 , 2])
    J = compute_cost(X, y, theta)
    print('Значение theta = [-1 ; 2]\nCost computed =', J)
    print('Ожидаемое значение функции стоимости (приблизительно): 54.24')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # 3. Проверяем градиентный спуск
    print('Градиентный спуск')
    alpha = 0.01  # шаг алгоритма
    iterations = 1500  # число итераций
    theta, J_hist = gradient_descent(X, y, theta, alpha, iterations)
    print('Theta found by gradient descent:')
    print(theta)
    print('Expected theta values (approx):')
    print('[-3.6303  1.1664]')
    # отображаем линейную регрессию на графике
    x0, x1 = min(x), max(x)
    y0 = theta[0] + theta[1] * x0
    y1 = theta[0] + theta[1] * x1
    plt.plot([x0, x1], [y0, y1], 'r')
    plt.legend(['Линейная регрессия', 'Обучающая выборка'])
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # 4. График функции стоимости
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in range(100):
        for j in range(100):
            t = np.array([theta0_vals[i] , theta1_vals[j]])
            J_vals[i, j] = compute_cost(X, y, t)

    t0 = np.outer(theta0_vals, np.ones(100))
    t1 = np.outer(np.ones(100), theta1_vals)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(t0, t1, J_vals)
    plt.show()
    plt.pause(5)
    # Контурный график
    plt.figure()
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.contour(t0, t1, J_vals, np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], 'rx')
    plt.show()
    plt.pause(5)

    input('Перейдите в терминал и нажмите Enter для завершения')


# Функция стоимости
def compute_cost(X, y, theta):
  m = y.shape[0]
  predictions = X.dot(theta)
  errors = predictions - y
  J = (1 / (2 * m)) * np.sum(errors**2)

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


if __name__ == '__main__':
    plt.ion()
    main()
    plt.clf()

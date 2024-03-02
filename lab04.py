import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt  


def main():
    ## 1. Загрузка данных
    data = np.loadtxt('ex2data1.txt', delimiter=',')  # чтение данных из текстового файла
    x = data[:, 0:2]  # первые две колонки - оценки по двум предметам
    y = data[:, 2]  # третья колонка - результат (поступил(1)/не поступил(0))
    print('Отображение обучающего набора данных на плоскости')
    plt.figure()
    plt.scatter(x[(y == 0), 0], x[(y == 0), 1], c='r', marker='o', label='Не поступил')
    plt.scatter(x[(y == 1), 0], x[(y == 1), 1], c='g', marker='o', label='Поступил')
    plt.xlabel('Предмет 1')
    plt.ylabel('Предмет 2')
    plt.legend()
    plt.title('Обучающая выборка')
    plt.show()
    plt.pause(1)
    print('Проверка функции сигмоиды')
    print('Значение g([0]):', sigmoid(np.array([0])), '  Ожидаемое значение: [0.5]')
    print('Значение g([-6, -1, 0, 1, 6]):', sigmoid(np.array([-6, -1, 0, 1, 6])), '  Ожидаемое значение(приблизительно): [0.00  0.27  0.50  0.73  1.00]')

    input('Перейдите в терминал и нажмите Enter для продолжения...')


    ## 2. Функция стоимости и градиент
    m = x.shape[0]
    X = np.hstack((np.ones((m, 1)), x))  # добавление к X фиктивного столбца единиц
    n = X.shape[1]
    initial_theta = np.zeros(n)  # Начальные тета примем нулевые
    cost = cost_function(initial_theta, X, y)  # Вычислим функцию стоимости для нулевого тета
    grad = gradient_function(initial_theta, X, y)  # Вычислим градиент для нулевого тета
    print('\nФункция стоимости для нулевого тета:', cost)
    print('Ожидаемое значение (приблизительно): 0.693')
    print('Градиент в нулевом тета:', grad)
    print('Ожидаемый градиент (приблизительно): [ -0.1000  -12.0092  -11.2628]')

    input('Перейдите в терминал и нажмите Enter для продолжения...')


    ## 3. Оптимизация
    # Используем функцию minimize из библиотеки scipy.optimize
    min_res = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y), jac=gradient_function)
    if not min_res.success:
        print('Ошибка оптимизации:', min_res.message)
    theta = min_res.x
    cost = cost_function(theta, X, y)
    print('\nФункция стоимости для наденного тета:', cost)
    print('Ожидаемое значение (приблизительно): 0.203')
    print('Тета:', theta)
    print('Ожидаемый тета (приблизительно): [ -25.161 0.206 0.201]')
    # Отобразим границу найденного решения на графике
    plt.figure()
    plt.scatter(x[(y == 0), 0], x[(y == 0), 1], c='r', marker='o', label='Не поступил')
    plt.scatter(x[(y == 1), 0], x[(y == 1), 1], c='g', marker='o', label='Поступил')
    plot_x = np.array([min(X[:, 2]) - 2,  max(X[:, 2]) + 2])
    plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
    plt.plot(plot_x, plot_y, label='Граница решения')
    plt.xlabel('Предмет 1')
    plt.ylabel('Предмет 2')
    plt.title('Граница решения')
    plt.legend()
    plt.show()
    plt.pause(1)

    input('Перейдите в терминал и нажмите Enter для продолжения...')

    
    ## 4. Предсказание поступления
    prob = sigmoid(np.array([1, 45, 85]).dot(theta))
    print('\nДля абитуриента с оценками 45 и 85 вероятность поступления:', prob)
    print('Ожидаемое значение: 0.775 ± 0.002')
    # Вычисление точности предсказания на обучающем наборе
    p = predict(X, theta)
    e = np.mean((p == y)) * 100
    print('Точность на обучении:', e)
    print('Ожидаемая точность (приблизительно): 89.0')


# Логистическая фукнция
def sigmoid(z):
    g = 1 / (1 + np.e**(-z))
    return g


# Функция стоимости логистической регрессии
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = -1 / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return J


def gradient_function(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    dth = (1 / m) * np.dot(X.T, (h - y))
    return dth.flatten()  # Функция minimize требует градиент в виде обычного массива


# Функция предсказания значений
def predict(X, theta):
    p = sigmoid(np.dot(X, theta))
    return p >= 0.5


if __name__ == '__main__':
    plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt  


def main():
    ## 1. Загрузка и отображение данных
    data = np.loadtxt('ex2data2.txt', delimiter=',')  # чтение данных из текстового файла
    x = data[:, 0:2]
    y = data[:, 2]
    print('Отображение обучающего набора данных')
    plot_data(x, y)
    
    input('Перейдите в терминал и нажмите Enter для продолжения...')


    ## 2. Регуляризованная логистическая регрессия
    X = map_feature(x[:, 0], x[:, 1])
    initial_theta = np.zeros(X.shape[1])
    lamb = 1
    cost = cost_function(initial_theta, X, y, lamb)
    grad = gradient_function(initial_theta, X, y, lamb)
    print('\nФункция стоимости для нулевого тета:', cost)
    print('Ожидаемое значение (приблизительно): 0.693')
    print('Градиент в нулевом тета (первые 5 элементов):', grad[:5])
    print('Ожидаемый градиент (приблизительно, первые 5 элементов): [ 0.0085  0.0188  0.0001  0.0503  0.0115 ]')
    initial_theta = np.ones(X.shape[1])
    lamb = 10
    cost = cost_function(initial_theta, X, y, lamb)
    grad = gradient_function(initial_theta, X, y, lamb)
    print('\nФункция стоимости для тестового тета:', cost)
    print('Ожидаемое значение (приблизительно): 3.16')
    print('Градиент в тестового тета (первые 5 элементов):', grad[:5])
    print('Ожидаемый градиент (приблизительно, первые 5 элементов): [ 0.3460  0.1614  0.1948  0.2269  0.0922 ]')

    input('Перейдите в терминал и нажмите Enter для продолжения...')


    ## 3. Оптимизация
    initial_theta = np.zeros(X.shape[1])
    lamb = 1
    min_res = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y, lamb), jac=gradient_function)
    if not min_res.success:
        print('Ошибка оптимизации:', min_res.message)
    theta = min_res.x
    cost = cost_function(theta, X, y, lamb)
    print('\nФункция стоимости для наденного тета:', cost)
    print('Ожидаемое значение (приблизительно): 0.529')
    print('Тета (первые 5 элементов):', theta[:5])
    print('Ожидаемый тета (приблизительно, первые 5 элементов): [ 1.273  0.625  1.177  -2.020  -0.913 ]')
    plot_boundary(x, y, theta, 'Граница решения (lambda=1)')

    input('Перейдите в терминал и нажмите Enter для продолжения...')


    ## 4. Оценка точности модели
    h1 = predict(map_feature([0.2], [0.2]), theta)
    print('Микрочип с тестами 0.2 и 0.2 модель относит к классу:', np.round(h1))
    print('Ожидаемое значение: 1 (годный)')
    h2 = predict(map_feature([1],[0.5]), theta)
    print('Микрочип с тестами 1 и 0.5 модель относит к классу:', np.round(h2))
    print('Ожидаемое значение: 0 (брак)')
    p = predict(X, theta)
    e = np.mean((p == y)) * 100
    print('Точность на обучении:', e)

    print('Ожидаемая точность (приблизительно): 83.1')


    ## 5. Влияние регуляризации
    # Обучим ту же самую модель без регуляризации (lamb = 0)
    initial_theta = np.zeros(X.shape[1])
    lamb = 0
    min_res = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y, lamb), jac=gradient_function)
    if not min_res.success:
        print('Ошибка оптимизации:', min_res.message)
    theta = min_res.x
    print('Построение границы решения модели (lamb = 0)')
    plot_boundary(x, y, theta, 'Граница решения (lambda=0)')
    # Обучим ту же самую модель с очень большой регуляризацией (lamb = 100)
    initial_theta = np.zeros(X.shape[1])
    lamb = 100
    min_res = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y, lamb), jac=gradient_function)
    if not min_res.success:
        print('Ошибка оптимизации:', min_res.message)
    theta = min_res.x
    print('Построение границы решения модели (lamb = 100)')
    plot_boundary(x, y, theta, 'Граница решения (lambda=100)')


# Отображение данных
def plot_data(x, y):
    plt.figure()
    plt.scatter(x[(y == 0), 0], x[(y == 0), 1], c='r', marker='o', label='Брак')
    plt.scatter(x[(y == 1), 0], x[(y == 1), 1], c='g', marker='o', label='Годный')
    plt.xlabel('Первый тест')
    plt.ylabel('Второй тест')
    plt.legend()
    plt.title('Обучающая выборка')
    plt.show()
    plt.pause(1)
    

# Логистическая функция
def sigmoid(z):
    g = 1 / (1 + np.e**(-z))
    return g


# Функция стоимости логистической регрессии
def cost_function(theta, X, y, lamb):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (1 / m) * (-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) + (lamb / (2 * m)) * np.sum(theta[1:] ** 2)
    return J


def gradient_function(theta, X, y, lamb):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    dth = (1 / m) * np.dot(X.T, (h - y)) + (lamb / m) * np.concatenate((np.zeros(1), theta[1:]))
    return dth.flatten()


# Функция предсказания значений
def predict(X, theta):
    p = sigmoid(np.dot(X, theta))
    return p >= 0.5


# Конструирование сложной модели для полинома 6 степени для двух переменных
def map_feature(X1, X2):
    out = np.ones((len(X1), 1))
    for i in range(1, 7):
        for j in range(i+1):
            col = np.power(X1, (i-j)) * np.power(X2, j)
            out = np.hstack((out, col.reshape(-1, 1)))
    return out


# Отображение границы решения
def plot_boundary(x, y, theta, title):
    plt.figure()
    plt.scatter(x[(y == 0), 0], x[(y == 0), 1], c='r', marker='o', label='Брак')
    plt.scatter(x[(y == 1), 0], x[(y == 1), 1], c='g', marker='o', label='Годный')
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = map_feature(u[i].reshape(1), v[j].reshape(1)).dot(theta)
    z = z.T
    plt.contour(u, v, z, levels=0, colors='blue')
    plt.xlabel('Тест 1')
    plt.ylabel('Тест 2')
    plt.title(title)
    plt.show()
    plt.pause(1)


if __name__ == '__main__':
    plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()

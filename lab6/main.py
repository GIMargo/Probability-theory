import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def lin(a, b):
    return lambda x: np.full_like(x, a)+np.full_like(x, b)*x \
           if isinstance(x, np.ndarray) else a + b * x

def absum(params, x, y):  # МНК формула (5)
    return np.sum(np.abs(y - params[0] - params[1] * x))

m_size = 20
ref = lin(2, 2) 
x = np.linspace(-1.8, 2, m_size)
y1 = ref(x) + norm.rvs(size=m_size) # эталонная зависимость
y2 = y1.copy() # выборка с возмущениями

y2[0] += 10
y2[19] -= 10

xm = np.mean(x)
d = np.mean(x*x) - xm**2 # знаменатель дроби

for y, name in zip([y1, y2], ['Выборка без возмущений',
                              'Выборка с возмущениями']):
    ym = np.mean(y) # среднее
    betahat1 = (np.mean(x*y) - ym*xm) / d  # оценка параметра beta_1
    mnk = (ym - xm*betahat1, betahat1) # МНК-оценки
    mnm = minimize(absum, [0, 1], args=(x, y), method='COBYLA').x
    plt.plot(x, ref(x), label='Эталон') 
    plt.plot(x, lin(*mnk)(x), label='МНК')
    plt.plot(x, lin(*mnm)(x), label='МНМ')
    plt.scatter(x, y, label='Выборка')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(name)
    print('МНК: ')
    print(mnk)
    print()
    print('МНМ: ')
    print(mnm)
    print()
    plt.show()

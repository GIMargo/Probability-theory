from scipy.stats import laplace
from scipy.stats import cauchy
from scipy.stats import uniform
from scipy.stats import poisson
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import sqrt, floor, exp, pi
import math
import numpy as np

fig, ax = plt.subplots(1, 3)
r = laplace.rvs(size=20)
x = np.linspace(min(laplace.ppf(0.01),min(r)), max(laplace.ppf(0.99),max(r)), 100)
ax[0].plot(x, laplace.cdf(x,0,sqrt(2)))
ax[0].hist(r, density=True, bins=floor(len(r)),histtype='step',cumulative=True)
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Лапласа, n=20')


r2 = laplace.rvs(size=60)
x2 = np.linspace(min(laplace.ppf(0.01),min(r2)), max(laplace.ppf(0.99),max(r2)), 100)
ax[1].plot(x2, laplace.cdf(x2,0,sqrt(2)))
ax[1].hist(r2, density=True, bins=floor(len(r2)),histtype='step',cumulative=True)
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Лапласа, n=60')

r3 = laplace.rvs(size=100)
x3 = np.linspace(min(laplace.ppf(0.01),min(r3)), max(laplace.ppf(0.99),max(r3)), 100)
ax[2].plot(x3, laplace.cdf(x3,0,sqrt(2)))
ax[2].hist(r3, density=True, bins=floor(len(r3)),histtype='step',cumulative=True)
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Лапласа, n=100')

#plt.show()#Лапласа

fig2, ay = plt.subplots(1, 3)
r = cauchy.rvs(size=20)
y = np.linspace(min(cauchy.ppf(0.01),min(r)), max(cauchy.ppf(0.99),max(r)), 100)
ay[0].plot(y, cauchy.cdf(y,0,1))
ay[0].hist(r, density=True, bins=floor(len(r)),histtype='step',cumulative=True)
ay[0].set_xlabel('x')
ay[0].set_title('Распределение Коши, n=20')

r2 = cauchy.rvs(size=60)
y2 = np.linspace(min(cauchy.ppf(0.01),min(r2)), max(cauchy.ppf(0.99),max(r2)), 100)
ay[1].plot(y2, cauchy.cdf(y2,0,1))
ay[1].hist(r2, density=True, bins=floor(len(r2)),histtype='step',cumulative=True)
ay[1].set_xlabel('x')
ay[1].set_title('Распределение Коши, n=60')

r3 = cauchy.rvs(size=100)
y3 = np.linspace(min(cauchy.ppf(0.01),min(r3)), max(cauchy.ppf(0.99),max(r3)), 100)
ay[2].plot(y3, cauchy.cdf(y3,0,1))
ay[2].hist(r3, density=True, bins=floor(len(r3)),histtype='step',cumulative=True)
ay[2].set_xlabel('x')
ay[2].set_title('Распределение Коши, n=100')

#plt.show()#Коши

fig3, ax = plt.subplots(1, 3)
r = uniform.rvs(-sqrt(3),2*sqrt(3),20)
x = np.linspace(-sqrt(3), sqrt(3), 100)
ax[0].plot(x, uniform.cdf(x,-sqrt(3),2*sqrt(3)))
ax[0].hist(r, density=True, bins=floor(len(r)),histtype='step',cumulative=True)
ax[0].set_xlabel('x')
ax[0].set_title('Равномерное распределение, n=20')

r2 = uniform.rvs(-sqrt(3),2*sqrt(3),60)
ax[1].plot(x, uniform.cdf(x,-sqrt(3),2*sqrt(3)))
ax[1].hist(r2, density=True, bins=floor(len(r2)),histtype='step',cumulative=True)
ax[1].set_xlabel('x')
ax[1].set_title('Равномерное распределение, n=60')

r3 = uniform.rvs(-sqrt(3),2*sqrt(3),100)
ax[2].plot(x, uniform.cdf(x,-sqrt(3),2*sqrt(3)))
ax[2].hist(r3, density=True, bins=floor(len(r3)),histtype='step',cumulative=True)
ax[2].set_xlabel('x')
ax[2].set_title('Равномерное распределение, n=100')

#plt.show()#Равномерное

fig4, ax = plt.subplots(1, 3)
r = poisson.rvs(10, size=20)
x = np.arange(min(poisson.ppf(0.01, 10),min(r)), max(poisson.ppf(0.99, 10),max(r)))
ax[0].plot(x, poisson.cdf(x,10))
ax[0].hist(r, density=True, bins=len(r),histtype='step',cumulative=True)
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Пуассона, n=20')

r2 = poisson.rvs(10, size=60)
x2 = np.arange(min(poisson.ppf(0.01, 10),min(r2)), max(poisson.ppf(0.99, 10),max(r2)))
ax[1].plot(x2, poisson.cdf(x2,10))
ax[1].hist(r2, density=True, bins=len(r2),histtype='step',cumulative=True)
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Пуассона, n=60')

r3 = poisson.rvs(10, size=100)
x3 = np.arange(min(poisson.ppf(0.01, 10),min(r3)), max(poisson.ppf(0.99, 10),max(r3)))
ax[2].plot(x3, poisson.cdf(x3,10))
ax[2].hist(r3, density=True, bins=len(r3),histtype='step',cumulative=True)
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Пуассона, n=100')

#plt.show()#Пуассона

fig5, ax = plt.subplots(1, 3)
r = norm.rvs(size=20)
x = np.linspace(min(norm.ppf(0.01),min(r)), max(norm.ppf(0.99),max(r)), 100)
ax[0].plot(x, norm.cdf(x,0,1))
ax[0].hist(r, density=True, bins=floor(len(r)),histtype='step',cumulative=True)
ax[0].set_xlabel('x')
ax[0].set_title('Нормальное распределение, n=20')

r2 = norm.rvs(size=60)
x2 = np.linspace(min(norm.ppf(0.01),min(r2)), max(norm.ppf(0.99),max(r2)), 100)
ax[1].plot(x2, norm.cdf(x2,0,1))
ax[1].hist(r2, density=True, bins=floor(len(r2)),histtype='step',cumulative=True)
ax[1].set_xlabel('x')
ax[1].set_title('Нормальное распределение, n=60')

r3 = norm.rvs(size=100)
x3 = np.linspace(min(norm.ppf(0.01),min(r3)), max(norm.ppf(0.99),max(r3)), 100)
ax[2].plot(x3, norm.cdf(x3,0,1))
ax[2].hist(r3, density=True, bins=floor(len(r3)),histtype='step',cumulative=True)
ax[2].set_xlabel('x')
ax[2].set_title('Нормальное распределение, n=100')

#plt.show()#Нормальное

def kernel_function(x: float):
    return exp(-x*x/2)/sqrt(2*pi)

def kernel_app(x: np.ndarray, r: np.ndarray, k):
    n = len(r)
    s = sqrt((r*r).sum()/n - (r.sum()/n)**2) #среднеквадратическое отклонение
    h = 1.06*s*(n**(-0.2))*k
    res = np.zeros_like(x)
    for i in range(len(x)):
        for u in r:
            res[i] += kernel_function((x[i] - u)/h)
        res[i] /= n*h
    return res

#Ядерные оценки
fig, ax = plt.subplots(1, 3)
rng = (-4, 4)
r = norm.rvs(size=20)
x = np.linspace(*rng, 100)
y1 = norm.pdf(x,0,1)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Нормальное распределение, n=20')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Нормальное распределение, n=20')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Нормальное распределение, n=20')

#plt.show()#Нормальное20

fig, ax = plt.subplots(1, 3)
r = norm.rvs(size=60)
x = np.linspace(*rng, 100)
y1 = norm.pdf(x,0,1)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Нормальное распределение, n=60')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Нормальное распределение, n=60')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Нормальное распределение, n=60')

#plt.show()#Нормальное60

fig, ax = plt.subplots(1, 3)
r = norm.rvs(size=100)
x = np.linspace(*rng, 100)
y1 = norm.pdf(x,0,1)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Нормальное распределение, n=100')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Нормальное распределение, n=100')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Нормальное распределение, n=100')

#plt.show()#Нормальное100

fig, ax = plt.subplots(1, 3)
r = laplace.rvs(size=20)
x = np.linspace(*rng, 100)
y1 = laplace.pdf(x,0,sqrt(2))
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Лапласа, n=20')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Лапласа, n=20')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Лапласа, n=20')

#plt.show()#Лапласа20

fig, ax = plt.subplots(1, 3)
r = laplace.rvs(size=60)
x = np.linspace(*rng, 100)
y1 = laplace.pdf(x,0,sqrt(2))
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Лапласа, n=60')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Лапласа, n=60')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Лапласа, n=60')

#plt.show()#Лапласа60

fig, ax = plt.subplots(1, 3)
r = laplace.rvs(size=100)
x = np.linspace(*rng, 100)
y1 = laplace.pdf(x,0,sqrt(2))
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Лапласа, n=100')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Лапласа, n=100')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Лапласа, n=100')

#plt.show()#Лапласа100

fig, ax = plt.subplots(1, 3)
r = cauchy.rvs(size=20)
x = np.linspace(*rng, 100)
y1 = cauchy.pdf(x,0,1)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Коши, n=20')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Коши, n=20')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Коши, n=20')

#plt.show()#Коши20

fig, ax = plt.subplots(1, 3)
r = cauchy.rvs(size=60)
x = np.linspace(*rng, 100)
y1 = cauchy.pdf(x,0,1)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Коши, n=60')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Коши, n=60')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Коши, n=60')

#plt.show()#Коши60

fig, ax = plt.subplots(1, 3)
r = cauchy.rvs(size=100)
x = np.linspace(*rng, 100)
y1 = cauchy.pdf(x,0,1)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Коши, n=100')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Коши, n=100')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Коши, n=100')

#plt.show()#Коши100

fig, ax = plt.subplots(1, 3)
r = uniform.rvs(size=20)
x = np.linspace(*rng, 100)
y1 = uniform.pdf(x,-sqrt(3),2*sqrt(3))
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Равномерное распределение, n=20')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Равномерное распределение, n=20')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Равномерное распределение, n=20')

#plt.show()#Равномерное20

fig, ax = plt.subplots(1, 3)
r = uniform.rvs(size=60)
x = np.linspace(*rng, 100)
y1 = uniform.pdf(x,-sqrt(3),2*sqrt(3))
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Равномерное распределение, n=60')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Равномерное распределение, n=60')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Равномерное распределение, n=60')

#plt.show()#Равномерное60

fig, ax = plt.subplots(1, 3)
r = uniform.rvs(size=100)
x = np.linspace(*rng, 100)
y1 = uniform.pdf(x,-sqrt(3),2*sqrt(3))
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Равномерное распределение, n=100')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Равномерное распределение, n=100')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Равномерное распределение, n=100')

#plt.show()#Равномерное100

fig, ax = plt.subplots(1, 3)
rng = (6, 14)
r = poisson.rvs(10, size=20)
x = np.linspace(*rng, 100)
y1 = poisson.pmf(np.floor(x), 10)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Пуассона, n=20')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Пуассона, n=20')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Пуассона, n=20')

#plt.show()#Пуассона20

fig, ax = plt.subplots(1, 3)
rng = (6, 14)
r = poisson.rvs(10, size=60)
x = np.linspace(*rng, 100)
y1 = poisson.pmf(np.floor(x), 10)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Пуассона, n=60')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Пуассона, n=60')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Пуассона, n=60')

#plt.show()#Пуассона60

fig, ax = plt.subplots(1, 3)
rng = (6, 14)
r = poisson.rvs(10, size=100)

x = np.linspace(*rng, 100)
y1 = poisson.pmf(np.floor(x), 10)
y2 = kernel_app(x, r, 0.5)
ax[0].plot(x, y1, label='плотность')
ax[0].plot(x, y2, label='оценка')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_title('Распределение Пуассона, n=100')

y3 = kernel_app(x, r, 1)
ax[1].plot(x, y1, label='плотность')
ax[1].plot(x, y3, label='оценка')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_title('Распределение Пуассона, n=100')

y4 = kernel_app(x, r, 2)
ax[2].plot(x, y1, label='плотность')
ax[2].plot(x, y4, label='оценка')
ax[2].legend()
ax[2].set_xlabel('x')
ax[2].set_title('Распределение Пуассона, n=100')

plt.show()#Пуассона100

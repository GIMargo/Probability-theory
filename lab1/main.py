# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from scipy.stats import laplace
from scipy.stats import cauchy
from scipy.stats import uniform
from scipy.stats import poisson
from scipy.stats import norm
import matplotlib.pyplot as plt
import hist
from math import sqrt, floor
import math
import numpy as np

fig, ax = plt.subplots(2, 3)
r = laplace.rvs(size=10)
x = np.linspace(min(laplace.ppf(0.01),min(r)), max(laplace.ppf(0.99),max(r)), 100)
ax[0][0].plot(x, laplace.pdf(x,0,sqrt(2)), label='теор.')
ax[0][0].set_title('Распределение Лапласа, n=10')
ax[0][0].hist(r, density=True, bins=floor(sqrt(len(r))),histtype='step',label='практ.')
ax[0][0].legend(loc='best', frameon=False)
ax[0][0].set_xlabel('x')
ax[0][0].set_ylabel('Функция плотности')
ax[0][0].legend()

ax[1][0].plot(x, laplace.cdf(x,0,sqrt(2)), label='теор.')
ax[1][0].hist(r, density=True, bins=floor(sqrt(len(r))),histtype='step',cumulative=True,label='практ.')
ax[1][0].legend(loc='best', frameon=False)
ax[1][0].set_xlabel('x')
ax[1][0].set_ylabel('Функция распределения')
ax[1][0].legend()

r2 = laplace.rvs(size=100)
x2 = np.linspace(min(laplace.ppf(0.01),min(r2)), max(laplace.ppf(0.99),max(r2)), 100)
ax[0][1].plot(x2, laplace.pdf(x2,0,sqrt(2)), label='теор.')
ax[0][1].set_title('Распределение Лапласа, n=100')
ax[0][1].hist(r2, density=True, bins=floor(sqrt(len(r2))),histtype='step',label='практ.')
ax[0][1].legend(loc='best', frameon=False)
ax[0][1].set_xlabel('x')
ax[0][1].set_ylabel('Функция плотности')
ax[0][1].legend()

ax[1][1].plot(x2, laplace.cdf(x2,0,sqrt(2)), label='теор.')
ax[1][1].hist(r2, density=True, bins=floor(sqrt(len(r2))),histtype='step',cumulative=True,label='практ.')
ax[1][1].legend(loc='best', frameon=False)
ax[1][1].set_xlabel('x')
ax[1][1].set_ylabel('Функция распределения')
ax[1][1].legend()

r3 = laplace.rvs(size=1000)
x3 = np.linspace(min(laplace.ppf(0.01),min(r3)), max(laplace.ppf(0.99),max(r3)), 100)
ax[0][2].plot(x3, laplace.pdf(x3,0,sqrt(2)), label='теор.')
ax[0][2].set_title('Распределение Лапласа, n=1000')
ax[0][2].hist(r3, density=True, bins=floor(sqrt(len(r3))),histtype='step',label='практ.')
ax[0][2].legend(loc='best', frameon=False)
ax[0][2].set_xlabel('x')
ax[0][2].set_ylabel('Функция плотности')
ax[0][2].legend()

ax[1][2].plot(x3, laplace.cdf(x3,0,sqrt(2)), label='теор.')
ax[1][2].hist(r3, density=True, bins=floor(sqrt(len(r3))),histtype='step',cumulative=True,label='практ.')
ax[1][2].legend(loc='best', frameon=False)
ax[1][2].set_xlabel('x')
ax[1][2].set_ylabel('Функция распределения')
ax[1][2].legend()

plt.show()#Лапласа

fig2, ay = plt.subplots(2, 3)
r = cauchy.rvs(size=10)
y = np.linspace(min(cauchy.ppf(0.01),min(r)), max(cauchy.ppf(0.99),max(r)), 100)
ay[0][0].plot(y, cauchy.pdf(y,0,1), label='теор.')
ay[0][0].set_title('Распределение Коши, n=10')
ay[0][0].hist(r, density=True, bins=floor(sqrt(len(r))),histtype='step',label='практ.')
ay[0][0].legend(loc='best', frameon=False)
ay[0][0].set_xlabel('x')
ay[0][0].set_ylabel('Функция плотности')
ay[0][0].legend()

ay[1][0].plot(y, cauchy.cdf(y,0,1), label='теор.')
ay[1][0].hist(r, density=True, bins=floor(sqrt(len(r))),histtype='step',cumulative=True,label='практ.')
ay[1][0].legend(loc='best', frameon=False)
ay[1][0].set_xlabel('x')
ay[1][0].set_ylabel('Функция распределения')
ay[1][0].legend()

r2 = cauchy.rvs(size=100)
y2 = np.linspace(min(cauchy.ppf(0.01),min(r2)), max(cauchy.ppf(0.99),max(r2)), 100)
ay[0][1].plot(y2, cauchy.pdf(y2,0,1), label='теор.')
ay[0][1].set_title('Распределение Коши, n=100')
ay[0][1].hist(r2, density=True, bins=floor(sqrt(len(r2))),histtype='step',label='практ.')
ay[0][1].legend(loc='best', frameon=False)
ay[0][1].set_xlabel('x')
ay[0][1].set_ylabel('Функция плотности')
ay[0][1].legend()

ay[1][1].plot(y2, cauchy.cdf(y2,0,1), label='теор.')
ay[1][1].hist(r2, density=True, bins=floor(sqrt(len(r2))),histtype='step',cumulative=True,label='практ.')
ay[1][1].legend(loc='best', frameon=False)
ay[1][1].set_xlabel('x')
ay[1][1].set_ylabel('Функция распределения')
ay[1][1].legend()

r3 = cauchy.rvs(size=1000)
y3 = np.linspace(min(cauchy.ppf(0.01),min(r3)), max(cauchy.ppf(0.99),max(r3)), 100)
ay[0][2].plot(y3, cauchy.pdf(y3,0,1), label='теор.')
ay[0][2].set_title('Распределение Коши, n=1000')
ay[0][2].hist(r3, density=True, bins=floor(sqrt(len(r3))),histtype='step',label='практ.')
ay[0][2].legend(loc='best', frameon=False)
ay[0][2].set_xlabel('x')
ay[0][2].set_ylabel('Функция плотности')
ay[0][2].legend()

ay[1][2].plot(y3, cauchy.cdf(y3,0,1), label='теор.')
ay[1][2].hist(r3, density=True, bins=floor(sqrt(len(r3))),histtype='step',cumulative=True,label='практ.')
ay[1][2].legend(loc='best', frameon=False)
ay[1][2].set_xlabel('x')
ay[1][2].set_ylabel('Функция распределения')
ay[1][2].legend()

plt.show()#Коши


fig3, ax = plt.subplots(2, 3)
r = uniform.rvs(-sqrt(3),2*sqrt(3),10)
x = np.linspace(-sqrt(3), sqrt(3), 100)
ax[0][0].plot(x, uniform.pdf(x,-sqrt(3),2*sqrt(3)), label='теор.')
ax[0][0].set_title('Равномерное распределение, n=10')
ax[0][0].hist(r, density=True, bins=floor(sqrt(len(r))),histtype='step',label='практ.')
ax[0][0].legend(loc='best', frameon=False)
ax[0][0].set_xlabel('x')
ax[0][0].set_ylabel('Функция плотности')
ax[0][0].legend()

ax[1][0].plot(x, uniform.cdf(x,-sqrt(3),2*sqrt(3)), label='теор.')
ax[1][0].hist(r, density=True, bins=floor(sqrt(len(r))),histtype='step',cumulative=True,label='практ.')
ax[1][0].legend(loc='best', frameon=False)
ax[1][0].set_xlabel('x')
ax[1][0].set_ylabel('Функция распределения')
ax[1][0].legend()

r2 = uniform.rvs(-sqrt(3),2*sqrt(3),100)
ax[0][1].plot(x, uniform.pdf(x,-sqrt(3),2*sqrt(3)), label='теор.')
ax[0][1].set_title('Равномерное распределение, n=100')
ax[0][1].hist(r2, density=True, bins=floor(sqrt(len(r2))),histtype='step',label='практ.')
ax[0][1].legend(loc='best', frameon=False)
ax[0][1].set_xlabel('x')
ax[0][1].set_ylabel('Функция плотности')
ax[0][1].legend()

ax[1][1].plot(x, uniform.cdf(x,-sqrt(3),2*sqrt(3)), label='теор.')
ax[1][1].hist(r2, density=True, bins=floor(sqrt(len(r2))),histtype='step',cumulative=True,label='практ.')
ax[1][1].legend(loc='best', frameon=False)
ax[1][1].set_xlabel('x')
ax[1][1].set_ylabel('Функция распределения')
ax[1][1].legend()

r3 = uniform.rvs(-sqrt(3),2*sqrt(3),1000)
ax[0][2].plot(x, uniform.pdf(x,-sqrt(3),2*sqrt(3)), label='теор.')
ax[0][2].set_title('Равномерное распределение, n=1000')
ax[0][2].hist(r3, density=True, bins=floor(sqrt(len(r3))),histtype='step',label='практ.')
ax[0][2].legend(loc='best', frameon=False)
ax[0][2].set_xlabel('x')
ax[0][2].set_ylabel('Функция плотности')
ax[0][2].legend()

ax[1][2].plot(x, uniform.cdf(x,-sqrt(3),2*sqrt(3)), label='теор.')
ax[1][2].hist(r3, density=True, bins=floor(sqrt(len(r3))),histtype='step',cumulative=True,label='практ.')
ax[1][2].legend(loc='best', frameon=False)
ax[1][2].set_xlabel('x')
ax[1][2].set_ylabel('Функция распределения')
ax[1][2].legend()

plt.show()#Равномерное

fig4, ax = plt.subplots(2, 3)
r = poisson.rvs(10, size=10)
x = np.arange(min(poisson.ppf(0.01, 10),min(r)), max(poisson.ppf(0.99, 10),max(r)))
ax[0][0].plot(x, poisson.pmf(x,10), label='теор.')
ax[0][0].set_title('Распределение Пуассона, n=10')
ax[0][0].hist(r, density=True, bins=(max(r)-min(r)+1),histtype='step',label='практ.')
ax[0][0].legend(loc='best', frameon=False)
ax[0][0].set_xlabel('x')
ax[0][0].set_ylabel('Функция плотности')
ax[0][0].legend()

ax[1][0].plot(x, poisson.cdf(x,10), label='теор.')
ax[1][0].hist(r, density=True, bins=(max(r)-min(r)+1),histtype='step',cumulative=True,label='практ.')
ax[1][0].legend(loc='best', frameon=False)
ax[1][0].set_xlabel('x')
ax[1][0].set_ylabel('Функция распределения')
ax[1][0].legend()

r2 = poisson.rvs(10, size=100)
x2 = np.arange(min(poisson.ppf(0.01, 10),min(r2)), max(poisson.ppf(0.99, 10),max(r2)))
ax[0][1].plot(x2, poisson.pmf(x2,10), label='теор.')
ax[0][1].set_title('Распределение Пуассона, n=100')
ax[0][1].hist(r2, density=True, bins=(max(r2)-min(r2)+1),histtype='step',label='практ.')
ax[0][1].legend(loc='best', frameon=False)
ax[0][1].set_xlabel('x')
ax[0][1].set_ylabel('Функция плотности')
ax[0][1].legend()

ax[1][1].plot(x2, poisson.cdf(x2,10), label='теор.')
ax[1][1].hist(r2, density=True, bins=(max(r2)-min(r2)+1),histtype='step',cumulative=True,label='практ.')
ax[1][1].legend(loc='best', frameon=False)
ax[1][1].set_xlabel('x')
ax[1][1].set_ylabel('Функция распределения')
ax[1][1].legend()

r3 = poisson.rvs(10, size=1000)
x3 = np.arange(min(poisson.ppf(0.01, 10),min(r3)), max(poisson.ppf(0.99, 10),max(r3)))
ax[0][2].plot(x3, poisson.pmf(x3,10), label='теор.')
ax[0][2].set_title('Распределение Пуассона, n=1000')
ax[0][2].hist(r3, density=True, bins=(max(r3)-min(r3)+1),histtype='step',label='практ.')
ax[0][2].legend(loc='best', frameon=False)
ax[0][2].set_xlabel('x')
ax[0][2].set_ylabel('Функция плотности')
ax[0][2].legend()

ax[1][2].plot(x3, poisson.cdf(x3,10), label='теор.')
ax[1][2].hist(r3, density=True, bins=(max(r3)-min(r3)+1),histtype='step',cumulative=True,label='практ.')
ax[1][2].legend(loc='best', frameon=False)
ax[1][2].set_xlabel('x')
ax[1][2].set_ylabel('Функция распределения')
ax[1][2].legend()

plt.show()#Пуассона

fig5, ax = plt.subplots(2, 3)
r = norm.rvs(size=10)
x = np.linspace(min(norm.ppf(0.01),min(r)), max(norm.ppf(0.99),max(r)), 100)
ax[0][0].plot(x, norm.pdf(x,0,1), label='теор.')
ax[0][0].set_title('Нормальное распределение, n=10')
ax[0][0].hist(r, density=True, bins=floor(sqrt(len(r))),histtype='step',label='практ.')
ax[0][0].legend(loc='best', frameon=False)
ax[0][0].set_xlabel('x')
ax[0][0].set_ylabel('Функция плотности')
ax[0][0].legend()

ax[1][0].plot(x, norm.cdf(x,0,1), label='теор.')
ax[1][0].hist(r, density=True, bins=floor(sqrt(len(r))),histtype='step',cumulative=True,label='практ.')
ax[1][0].legend(loc='best', frameon=False)
ax[1][0].set_xlabel('x')
ax[1][0].set_ylabel('Функция распределения')
ax[1][0].legend()

r2 = norm.rvs(size=100)
x2 = np.linspace(min(norm.ppf(0.01),min(r2)), max(norm.ppf(0.99),max(r2)), 100)
ax[0][1].plot(x2, norm.pdf(x2,0,1), label='теор.')
ax[0][1].set_title('Нормальное распределение, n=100')
ax[0][1].hist(r2, density=True, bins=floor(sqrt(len(r2))),histtype='step',label='практ.')
ax[0][1].legend(loc='best', frameon=False)
ax[0][1].set_xlabel('x')
ax[0][1].set_ylabel('Функция плотности')
ax[0][1].legend()

ax[1][1].plot(x2, norm.cdf(x2,0,1), label='теор.')
ax[1][1].hist(r2, density=True, bins=floor(sqrt(len(r2))),histtype='step',cumulative=True,label='практ.')
ax[1][1].legend(loc='best', frameon=False)
ax[1][1].set_xlabel('x')
ax[1][1].set_ylabel('Функция распределения')
ax[1][1].legend()

r3 = norm.rvs(size=1000)
x3 = np.linspace(min(norm.ppf(0.01),min(r3)), max(norm.ppf(0.99),max(r3)), 100)
ax[0][2].plot(x3, norm.pdf(x3,0,1), label='теор.')
ax[0][2].set_title('Нормальное распределение, n=1000')
ax[0][2].hist(r3, density=True, bins=floor(sqrt(len(r3))),histtype='step',label='практ.')
ax[0][2].legend(loc='best', frameon=False)
ax[0][2].set_xlabel('x')
ax[0][2].set_ylabel('Функция плотности')
ax[0][2].legend()

ax[1][2].plot(x3, norm.cdf(x3,0,1), label='теор.')
ax[1][2].hist(r3, density=True, bins=floor(sqrt(len(r3))),histtype='step',cumulative=True,label='практ.')
ax[1][2].legend(loc='best', frameon=False)
ax[1][2].set_xlabel('x')
ax[1][2].set_ylabel('Функция распределения')
ax[1][2].legend()

plt.show()#Нормальное
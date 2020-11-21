import scipy.stats as sps
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

fig, ax = plt.subplots(1, 1)
#нормальное 20
sample = sps.norm.rvs(size=20, loc=0, scale=1)
#нормальное 100
sample2 = sps.norm.rvs(size=100, loc=0, scale=1)
ax.boxplot([sample, sample2], vert=False, labels=['20', '100'])
ax.set_xlabel('x')
ax.set_ylabel('n')
ax.set_title('Нормальное распределение')
plt.show()

fig2, ax = plt.subplots(1, 1)
#равномерное 20
sample = sps.uniform.rvs(size=20, loc=-sqrt(3), scale=2*sqrt(3))
#равномерное 100
sample2 = sps.uniform.rvs(size=100, loc=-sqrt(3), scale=2*sqrt(3))
ax.boxplot([sample, sample2], vert=False, labels=['20', '100'])
ax.set_xlabel('x')
ax.set_ylabel('n')
ax.set_title('Равномерное распределение')
plt.show()

fig3, ax = plt.subplots(1, 1)
#Коши 20
sample = sps.cauchy.rvs(size=20, loc=0, scale=1)
#Коши 100
sample2 = sps.cauchy.rvs(size=100, loc=0, scale=1)
ax.boxplot([sample, sample2], vert=False, labels=['20', '100'])
ax.set_xlabel('x')
ax.set_ylabel('n')
ax.set_title('Распределение Коши')
plt.show()

fig4, ax = plt.subplots(1, 1)
#Лапласа 20
sample = sps.laplace.rvs(size=20, loc=0, scale=sqrt(2))
#Лапласа 100
sample2 = sps.laplace.rvs(size=100, loc=0, scale=sqrt(2))
ax.boxplot([sample, sample2], vert=False, labels=['20', '100'])
ax.set_xlabel('x')
ax.set_ylabel('n')
ax.set_title('Распределение Лапласа')
plt.show()

fig5, ax = plt.subplots(1, 1)
#Пуассона 20
sample = sps.poisson.rvs(size=20, mu=10)
#Пуассона 100
sample2 = sps.poisson.rvs(size=100, mu=10)
ax.boxplot([sample, sample2], vert=False, labels=['20', '100'])
ax.set_xlabel('x')
ax.set_ylabel('n')
ax.set_title('Распределение Пуассона')
plt.show()

# Доля выбросов (нормальное распределение)
i=0
count = 0
while i<1000:
    sample = sps.norm.rvs(size=20, loc=0, scale=1)
    sample.sort()
    q1=sample[4]
    q3=sample[14]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (нормальное): %.5f'%(count/(1000*20)))

i=0
count=0
while i<1000:
    sample = sps.norm.rvs(size=100, loc=0, scale=1)
    sample.sort()
    q1=sample[24]
    q3=sample[74]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (нормальное): %.5f'%(count/(1000*100)))
print()

# Доля выбросов (равномерное распределение)
i=0
count = 0
while i<1000:
    sample = sps.uniform.rvs(size=20, loc=-sqrt(3), scale=2*sqrt(3))
    sample.sort()
    q1=sample[4]
    q3=sample[14]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (равномерное): %.5f'%(count/(1000*20)))

i=0
count=0
while i<1000:
    sample = sps.uniform.rvs(size=100, loc=-sqrt(3), scale=2*sqrt(3))
    sample.sort()
    q1=sample[24]
    q3=sample[74]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (равномерное): %.5f'%(count/(1000*100)))
print()

# Доля выбросов (распределение Лапласа)
i=0
count = 0
while i<1000:
    sample = sps.laplace.rvs(size=20, loc=0, scale=sqrt(2))
    sample.sort()
    q1=sample[4]
    q3=sample[14]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (Лаплас): %.5f'%(count/(1000*20)))

i=0
count=0
while i<1000:
    sample = sps.laplace.rvs(size=100, loc=0, scale=sqrt(2))
    sample.sort()
    q1=sample[24]
    q3=sample[74]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (Лаплас): %.5f'%(count/(1000*100)))
print()

# Доля выбросов (распределение Коши)
i=0
count = 0
while i<1000:
    sample = sps.cauchy.rvs(size=20, loc=0, scale=1)
    sample.sort()
    q1=sample[4]
    q3=sample[14]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (Коши): %.5f'%(count/(1000*20)))

i=0
count=0
while i<1000:
    sample = sps.cauchy.rvs(size=100, loc=0, scale=1)
    sample.sort()
    q1=sample[24]
    q3=sample[74]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (Коши): %.5f'%(count/(1000*100)))
print()

# Доля выбросов (распределение Пуассона)
i=0
count = 0
while i<1000:
    sample = sps.poisson.rvs(size=20, mu=10)
    sample.sort()
    q1=sample[4]
    q3=sample[14]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (Пуассон): %.5f'%(count/(1000*20)))

i=0
count=0
while i<1000:
    sample = sps.poisson.rvs(size=100, mu=10)
    sample.sort()
    q1=sample[24]
    q3=sample[74]
    x1=q1-1.5*(q3-q1)
    x2=q3+1.5*(q3-q1)

    for x in sample:
        if (x>x2 or x<x1):
            count=count+1

    i=i+1

print('Среднее число выбросов (Пуассон): %.5f'%(count/(1000*100)))
print()

from scipy.stats import multivariate_normal as mvnd
from math import sqrt, floor, exp, pi
import math
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms


def pearson(x, y):
    n = len(x)
    xm = sum(x) / n
    ym = sum(x) / n
    cov = sum([(x[i] - xm)*(y[i] - ym) for i in range(n)])
    ss = sqrt(sum([(x[i]-xm)**2 for i in range(n)]) * \
              sum([(y[i]-ym)**2 for i in range(n)]))
    return cov / ss

def quadrant(x, y):
    xm = np.median(x)
    ym = np.median(y)
    sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0
    return sum(map(lambda a: sign((a[0]-xm)*(a[1]-ym)), zip(x, y))) / len(x)

def spearman(x, y):
    sx = sorted(x)
    sy = sorted(y)
    rx = [sx.index(a) for a in x]
    ry = [sy.index(a) for a in y]
    return pearson(rx, ry)

def plotEllipse(x, y, ax, sigmas):
    r = pearson(x, y)
    ellipse = Ellipse((0, 0),
                      width=sqrt(1 + r),
                      height=sqrt(1 - r),
                      facecolor='none',
                      edgecolor='orange')
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    sx = sqrt(sum(map(lambda a: (a - xm)**2, x)) / len(x))
    sy = sqrt(sum(map(lambda a: (a - ym)**2, y)) / len(y))
    tr = transforms.Affine2D() \
         .rotate_deg(45) \
         .scale(sx * sigmas, sy * sigmas) \
         .translate(xm, ym)
    ellipse.set_transform(tr + ax.transData)
    ax.add_patch(ellipse)
    ax.scatter(x, y, s=0.9)

#Размер 20, р=0
m_size=20
rho=0
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print('Размер выборки: 20, р=0')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 20, р=0.5
m_size=20
rho=0.5
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print()  
print('Размер выборки: 20, р=0.5')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 20, р=0.9
m_size=20
rho=0.9
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print()  
print('Размер выборки: 20, р=0.9')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 60, р=0
m_size=60
rho=0
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print()  
print('Размер выборки: 60, р=0')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 60, р=0.5
m_size=60
rho=0.5
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print()  
print('Размер выборки: 60, р=0.5')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 60, р=0.9
m_size=60
rho=0.9
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print()  
print('Размер выборки: 60, р=0.9')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 100, р=0
m_size=100
rho=0
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print()  
print('Размер выборки: 100, р=0')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 100, р=0.5
m_size=100
rho=0.5
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print()  
print('Размер выборки: 100, р=0.5')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 100, р=0.9
m_size=100
rho=0.9
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, rho], [rho, 1]], size=m_size)

   x=[sample[0][0]]
   y=[sample[0][1]]

   for i in range(1,m_size):
       x.append(sample[i][0])

   for i in range(1,m_size):
       y.append(sample[i][1])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1
   
print()  
print('Размер выборки: 100, р=0.9')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Смесь, размер: 20
m_size=20
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, 0.9], [0.9, 1]], size=int(m_size*0.9))
   sample = np.append(sample,mvnd.rvs((0, 0), [[100, -90], [-90, 100]], size=int(m_size*0.1)))
   x=[sample[0]]
   y=[sample[1]]

   for i in range(2,m_size-1):
       if i % 2 == 0:
          x.append(sample[i])

   for i in range(3,m_size):
       if i % 2 != 0:
          y.append(sample[i])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1

print()  
print('Cмесь, размер=20')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Смесь, размер: 60
m_size=60
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, 0.9], [0.9, 1]], size=int(m_size*0.9))
   sample = np.append(sample,mvnd.rvs((0, 0), [[100, -90], [-90, 100]], size=int(m_size*0.1)))
   x=[sample[0]]
   y=[sample[1]]

   for i in range(2,m_size-1):
       if i % 2 == 0:
          x.append(sample[i])

   for i in range(3,m_size):
       if i % 2 != 0:
          y.append(sample[i])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1

print()  
print('Cмесь, размер=60')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Смесь, размер: 100
m_size=100
j=0
sumpearson=0
sumspearman=0
sumquadrant=0
quadrpear=0
quadrsper=0
quadrquadr=0
while j<1000:
   sample = mvnd.rvs((0, 0), [[1, 0.9], [0.9, 1]], size=int(m_size*0.9))
   sample = np.append(sample,mvnd.rvs((0, 0), [[100, -90], [-90, 100]], size=int(m_size*0.1)))
   x=[sample[0]]
   y=[sample[1]]

   for i in range(2,m_size-1):
       if i % 2 == 0:
          x.append(sample[i])

   for i in range(3,m_size):
       if i % 2 != 0:
          y.append(sample[i])

   p=pearson(x,y)
   s=spearman(x,y)
   q=quadrant(x,y)
   sumpearson+=p
   quadrpear+=p**2
   sumspearman+=s
   quadrsper+=s**2
   sumquadrant+=q
   quadrquadr+=q**2
   j+=1

print()  
print('Cмесь, размер=100')
print()
print('Среднее коэффициентов Пирсона: %.5f' %(sumpearson/1000))
print('Среднее коэффициентов Спирмена: %.5f' %(sumspearman/1000))
print('Среднее квадрантных коэффициентов корреляции: %.5f' %(sumquadrant/1000))
print()
print('Среднее квадратов коэффициентов Пирсона: %.5f' %(quadrpear/1000))
print('Среднее квадратов коэффициентов Спирмена: %.5f' %(quadrsper/1000))
print('Среднее квадратов квадрантных коэффициентов корреляции: %.5f' %(quadrquadr/1000))
print()
print('Дисперсия коэффициентов Пирсона: %.5f' %((quadrpear/1000)-(sumpearson/1000)**2))
print('Дисперсия коэффициентов Спирмена: %.5f' %((quadrsper/1000)-(sumspearman/1000)**2))
print('Дисперсия квадрантных коэффициентов корреляции: %.5f' %((quadrquadr/1000)-(sumquadrant/1000)**2))

#Размер 20
m_size=20
rhos = [0, 0.5, 0.9]
    
fig, ax = plt.subplots(1, 3)
for j in range(0,3):
    a = ax[j]
    sample = mvnd.rvs((0, 0), [[1, rhos[j]], [rhos[j], 1]], size=m_size)
    x=[sample[0][0]]
    y=[sample[0][1]]

    for i in range(1,m_size):
        x.append(sample[i][0])

    for i in range(1,m_size):
        y.append(sample[i][1])
        
    a.set_title(f'n = {m_size}, rho = {rhos[j]}')
    a.axis('equal')
    plotEllipse(x, y, a, 3)

#Размер 60
m_size=60
rhos = [0, 0.5, 0.9]
    
fig2, ax = plt.subplots(1, 3)
for j in range(0,3):
    a = ax[j]
    sample = mvnd.rvs((0, 0), [[1, rhos[j]], [rhos[j], 1]], size=m_size)
    x=[sample[0][0]]
    y=[sample[0][1]]

    for i in range(1,m_size):
        x.append(sample[i][0])

    for i in range(1,m_size):
        y.append(sample[i][1])
        
    a.set_title(f'n = {m_size}, rho = {rhos[j]}')
    a.axis('equal')
    plotEllipse(x, y, a, 3)

#Размер 100
m_size=100
fig3, ax = plt.subplots(1, 3)
for j in range(0,3):
    a = ax[j]
    sample = mvnd.rvs((0, 0), [[1, rhos[j]], [rhos[j], 1]], size=m_size)
    x=[sample[0][0]]
    y=[sample[0][1]]

    for i in range(1,m_size):
        x.append(sample[i][0])

    for i in range(1,m_size):
        y.append(sample[i][1])
        
    a.set_title(f'n = {m_size}, rho = {rhos[j]}')
    a.axis('equal')
    plotEllipse(x, y, a, 3)
    
plt.show()

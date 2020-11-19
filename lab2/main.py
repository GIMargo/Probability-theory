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

#нормальное 10
s =10
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.norm.rvs(size=s, loc=0, scale=1)
    sample.sort()
    #print('Выборочное среднее: %.3f'%sample.mean())
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    #print('Выборочная медиана: %.3f'%((sample[half-1]+sample[half])/2))
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    #print('Полусумма экстремальных выборочных элементов: %.3f'%((sample[0]+sample[s-1])/2))
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    #print('Полусумма квартилей: %.3f'%((sample[int(s/4)]+sample[int(s*3/4)])/2))
    ch4=ch4+(sample[int(s/4)]+sample[int(s*3/4)])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)]+sample[int(s*3/4)])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    #print('Усечённое среднее: %.3f'%(sum/half))
    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()


#нормальное 100
s =100
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.norm.rvs(size=s, loc=0, scale=1)
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()



#нормальное 1000
s =1000
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.norm.rvs(size=s, loc=0, scale=1)
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#равномерное 10
s =10
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.uniform.rvs(size=s, loc=-sqrt(3), scale=2 * sqrt(3))
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#равномерное 100
s =100
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.uniform.rvs(size=s, loc=-sqrt(3), scale=2 * sqrt(3))
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#равномерное 1000
s =1000
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.uniform.rvs(size=s, loc=-sqrt(3), scale=2*sqrt(3))
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Коши 10
s =10
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.cauchy.rvs(size=s, loc=0, scale=1)
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Коши 100
s =100
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.cauchy.rvs(size=s, loc=0, scale=1)
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Коши 1000
s =1000
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.cauchy.rvs(size=s, loc=0, scale=1)
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Лапласа 10
s =10
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.laplace.rvs(size=s, loc=0, scale=sqrt(2))
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Лапласа 100
s =100
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.laplace.rvs(size=s, loc=0, scale=sqrt(2))
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Лапласа 1000
s =1000
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.laplace.rvs(size=s, loc=0, scale=sqrt(2))
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Пуассона 10
s =10
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.poisson.rvs(size=s, mu=10)
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Пуассона 100
s =100
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.poisson.rvs(size=s, mu=10)
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()

#Пуассона 1000
s =1000
half=int(s/2)
ch1=0
ch2=0
ch3=0
ch4=0
ch5=0
sqrch1=0
sqrch2=0
sqrch3=0
sqrch4=0
sqrch5=0
j=0
while j < 1000:
    sample = sps.poisson.rvs(size=s, mu=10)
    sample.sort()
    ch1=ch1+sample.mean()
    sqrch1=sqrch1+(sample.mean())**2
    ch2=ch2+(sample[half-1]+sample[half])/2
    sqrch2 = sqrch2 + ((sample[half-1]+sample[half])/2) ** 2
    ch3=ch3+(sample[0]+sample[s-1])/2
    sqrch3 = sqrch3 + ((sample[0]+sample[s-1])/2) ** 2
    ch4=ch4+(sample[int(s/4)-1]+sample[int(s*3/4)-1])/2
    sqrch4 = sqrch4 + ((sample[int(s/4)-1]+sample[int(s*3/4)-1])/2) ** 2

    sum=0
    i = int(s/4)
    while i < s-int(s/4):
        sum=sum+sample[i]
        i = i + 1

    ch5=ch5+(sum/half)
    sqrch5 = sqrch5 + ((sum/half)) ** 2
    j = j + 1
print('Выборочное среднее: %.5f'%(ch1/1000))
print('Оценка дисперсии выборочное среднее: %.5f'%((sqrch1/1000)-(ch1/1000)**2))
print('Выборочная медиана: %.5f' %(ch2/1000))
print('Оценка дисперсии выборочная медиана: %.5f' %((sqrch2/1000)-(ch2/1000)**2))
print('Полусумма экстремальных выборочных элементов: %.5f'%(ch3/1000))
print('Оценка дисперсии полусумма экстремальных выборочных элементов: %.5f'%((sqrch3/1000)-(ch3/1000)**2))
print('Полусумма квартилей: %.5f'%(ch4/1000))
print('Оценка дисперсии полусумма квартилей: %.5f'%((sqrch4/1000)-(ch4/1000)**2))
print('Усечённое среднее: %.5f'%(ch5/1000))
print('Оценка дисперсии усечённое среднее: %.5f'%((sqrch5/1000)-(ch5/1000)**2))
print()


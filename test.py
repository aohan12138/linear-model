import  matplotlib.pyplot as plt
import random
import matplotlib
 

#生成数据
def data():
    x = range(10)
    y = [(2*s+4) for s in x]
    for i in range(10):
        y[i] = y[i]+random.randint(0,8)-4
    return x, y

#用小批量梯度下降算法进行迭代
def MBGD(x,y):
    error0 = 0
    error1 = 0
    n = 0
    m = len(x)
    esp = 1e-6
    step_size = 0.01  #选择合理的步长
    a = random.randint(0,10)  #给a，b赋初始值
    b = random.randint(0,10)
    while True:
        trainList = []
        for i in range(5):  #创建随机的批量
            trainList.append(random.randint(0,m-1))
 
        for i in range(5):  #对数据进行迭代计算
            s = trainList[i]
            sum0 = a*x[s]+b-y[s]
            sum1 = (a*x[s]+b-y[s])*x[s]
            error1 = error1+(a*x[s]+b-y[s])**2
        a = a - sum1*step_size/m
        b = b - sum0*step_size/m
        print('a=%f,b=%f,error=%f'%(a,b,error1))
 
        if error1-error0<esp:
            break
 
        n = n+1
        if n>500:
            break
    return a, b
'''
MBGD的测试方法
    x,y = data()
    a,b = MBGD(x,y)
    X = range(len(x))
    Y = [(a*i+b) for i in X]
 
    plt.scatter(x,y,color='red')
    plt.plot(X,Y,color='blue')
    plt.show()
'''
#使用随机梯度下降训练
def SGD(x,y):
    error0 = 0
    step_size = 0.001
    esp = 1e-6
    #a = random.randint(0,4)
    #b = random.randint(0,8)
    a = 1.2  #将给a，b随机赋初始值
    b = 3.5
    m = len(x)
    n = 0
    while True:
        i = random.randint(0,m-1)
        print(i)
        sum0 = a * x[i] + b - y[i]
        sum1 = (a * x[i] + b - y[i])*x[i]
        error1 = (a * x[i] + b - y[i])**2  #计算模型和结果的误差
 
        a = a - sum1*step_size/m
        b = b - sum0*step_size/m
        print('a=%f,b=%f,error=%f'%(a,b,error1))
 
        if abs(error1-error0)<esp:  #误差很小，可以终止迭代
            break
        error0 = error1
        n = n+1
        if n%20==0:
            print('第%d次迭代'%n)
        if (n>500):
            break
    return a,b
'''
SGD的测试方法
    x,y = data()
    a,b = SGD(x,y)
    X = range(10)
    Y = [(a*i+b) for i in X]
 
    plt.scatter(x,y,color='red')
    plt.plot(X,Y)
    plt.show()
'''
#使用梯度下降进行训练
def diedai(x,y):
    flag = True
    a = random.randint(0,5)
    b = random.randint(0,10)
    m = len(x)
    arf = 0.005 #学习率
    n = 0
    sum1 = 0
    sum2 = 0
    exp = 0.000001
    error0 = 0
    error1 = 0
    while flag:
 
        for i in range(m):  #计算对应的偏导数
            sum1 = a*x[i]+b-y[i]
            sum2 = (a*x[i]+b-y[i])*x[i]
            error1 = (a*x[i]+b-y[i])**2
        a = a - sum2*arf/m  #对a，b进行更新
        b = b - sum1*arf/m
 
        if abs(error1-error0)<exp: #计算误差
            break
        error0 = error1
        print('a=%f,b=%f,error=%f' % (a, b, error1))
 
        if n > 500:
            #flag = False
            break
        n += 1
        if n % 10 == 0:
            print('第%d次迭代:a=%f,b=%f' % (n, a, b))
    return a,b
 
#使用最小二乘法计算结果
def calculation(x, y):
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    n = len(x)
    for i in range(n):
        c1 += x[i]*y[i]
        c2 += x[i]*x[i]
        c3 += x[i]
        c4 += y[i]
    a = (n*c1-c3*c4) /( n*c2-c3*c3)  #利用公式计算a, b
    b = (c2*c4-c3*c1) / (n*c2-c3*c3)
    return a, b
def GD():
     eta =0.1
     n_iterations =1000
     m =100 #样本数量
     X= 2*np.random.rand(100,1)  #产生一百个[0，2]的随机数
     y =4+ 3*X +np.random.rand(100,1)
       # np.c_表示按列操作拼接，np.r_表示按行操作拼
     X_b = np.c_[np.ones((100, 1)), X]     # x0 = 1
     theta = np.random.randn(2,1)
     for iteration in range(n_iterations):
         gradients =2/m*X_b.T.dot(X_b.dot(theta) - y)
         theta =theta - eta*gradients
         print(theta)
         _X =[X[0],X[1]]
         _Y =[theta[0]+theta[1]*x for x in _X]
         plt.xlabel('x变量')
         plt.ylabel('y变量')
         plt.title("函数")
         matplotlib.rcParams['font.sans-serif']=['SimHei']
         plt.scatter(X,y,color ='red',label = '数据')
         plt.plot(_X,_Y,'blue',label='拟合曲线')
         plt.show()
 
if __name__ == '__main__':
    x,y = data()
 
    a1,b1 = diedai(x,y)
    X1 = range(10)
    Y1 = [(a1*s+b1) for s in X1]
    print('梯度下降y=%fX+%f'%(a1,b1))
 
    a2,b2 = calculation(x,y)
    X2 = range(10)
    Y2 = [(a2*s+b2) for s in X2]
    print('最小二乘法y=%fX+%f'%(a2,b2))
 
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'] #设置字体中文，防止乱码
    plt.scatter(x, y, color = 'red',label = '数据')
    plt.plot(X1, Y1, color = 'blue',label = '梯度下降')
    plt.plot(X2, Y2, color = 'green',label = '最小二乘法')
    plt.legend()
    plt.show()
#梯度下降
#https://blog.csdn.net/cy776719526/article/details/80367392 来源

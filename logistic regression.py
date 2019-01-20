import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv(r'C:\Users\Administrator\Desktop\multivariate-linear-regression-master\loan_pred.csv')
print(data.shape)
print(data.head())

#删除所有缺少值的行
data.dropna(inplace=True)


#i导入库标签编码器
from sklearn.preprocessing import LabelEncoder
#创建一个包含分类谓词的列表
cat_var =['Gender','Married','Education','Self_Employed','Loan_Status']
#启动LabelEncodetr
le = LabelEncoder() 
#for循环将分类值转换为数值
for n in cat_var:
    data[n] = le.fit_transform(data[n])
print(data.head())
#随后检查谓词的类型
print(data.dtypes)

#将变量赋给数组
LoanAmount = data['LoanAmount'].values
Credit_History = data['Credit_History'].values
Loan_Status = data['Loan_Status'].values	


# 将分数绘制成散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(LoanAmount, Credit_History, Loan_Status, color='#ef1234')
plt.show()

#生成参数
m =len(LoanAmount)
x0=np.ones(m)
X= np.array([x0,LoanAmount,Credit_History]).T
#最初的系数
B = np.array([0,0,0])
Y = np.array(Loan_Status)
alpha =0.0001
#定义损失函数
def cost_function(X,Y,B):
    m = len(Y)
    J=np.sum((X.dot(B)-Y)**2)/(2*m)
    return J
inital_cost = cost_function(X,Y,B)
print("Inital Cost:")
print(inital_cost)

def gradient_descent(X,Y,B,alpha,iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # 假设值
        h = X.dot(B)
        #差异b/w假设和实际Y
        loss = h - Y
        #梯度计算
        gradient = X.T.dot(loss) / m
        # 用梯度法改变B的值
        B = B - alpha * gradient
        #新的成本价值
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history
#进行100 次的迭代
newB, cost_history = gradient_descent(X, Y, B, alpha, 100)  
# New Values of B
print("New Coefficients")
print(newB)

# Final Cost of new B
print("Final Cost")
print(cost_history[-1])

# 评估模型- RMSE . .
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

# 模型评估- R2评分
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = X.dot(newB)

print("RMSE")
print(rmse(Y, Y_pred))
print("R2 Score")
print(r2_score(Y, Y_pred))

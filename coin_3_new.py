import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(0)
'''
使用介绍： 使用前需要添加matplotlib包
在ThreeCoinMode类中 
构造函数中n_epoch是迭代次数
run_three_coins_model方法中的参数值是既定参数值
get_relusts方法是产生样本的方法，参数n是样本个数

__init_params中可以自定义初始值或者随机生成初始值
fit方法中不断调用E_Step方法和M_Step方法 进行问题的求解

'''



class ThreeCoinsMode():
    def __init__(self, n_epoch=10):
        """
        n_epoch 迭代次数
        """
        self.n_epoch = n_epoch
        self.params = {'s1': None, 's2': None, 'pi': None, 'p':None, 'q':None, 'mu': None, 'mu2': None} #迭代初始参数
        self.ele = {'s1': [1], 's2': [1], 'pi': [1], 'p': [1], 'q': [1]}#作图参数

    def __init_params(self, n):
        """
        对参数初始化操作
         n: 观测样本个数

        """
        '''
        self.params = {
                       's1': np.random.rand(1),
                       's2': np.random.rand(1),
                       'pi': np.random.rand(1),
                       'p': np.random.rand(1),
                       'q': np.random.rand(1),
                       'mu': np.random.rand(n),
                       'mu2': np.random.rand(n)}    
                       #随机生成初始值
        '''
        # 自定义初始值
        self.params = {
                       's1': [0.3],
                       's2': [0.1],
                       'pi': [0.2],
                       'p': [0.1],
                       'q': [0.1],
                       'mu2': np.random.rand(n),  #u初始值没什么影响，第一个E-Step会直接求解u值
                       'mu': np.random.rand(n)}

        self.ele = {
            's1':[0 for x in range(0, self.n_epoch+1)],
            's2': [0 for x in range(0, self.n_epoch+1)],
            'pi': [0 for x in range(0, self.n_epoch+1)],
            'p': [0 for x in range(0, self.n_epoch+1)],
            'q': [0 for x in range(0, self.n_epoch+1)],
        }

    def E_step(self, y, n):
        """
        更新隐变量u
        y 样本
        n 样本个数

        """
        pi = self.params['pi'][0]  #只有一个数值 所以读[0] 不然读出来是一个列表
        p = self.params['p'][0]
        q = self.params['q'][0]
        s1 = self.params['s1'][0]
        s2 = self.params['s2'][0]

        for i in range(n):
            self.params['mu'][i] = (s1 * pow(pi, y[i]) * pow(1-pi, 1-y[i])) / \
                                   (s1 * pow(pi, y[i]) * pow(1-pi, 1-y[i]) + s2 * pow(p, y[i]) * pow(1-p, 1-y[i])
                                    + (1-s1-s2) * pow(q,y[i])*pow(1-q,1-y[i]))
            self.params['mu2'][i] = (s2 * pow(p, y[i]) * pow(1-p, 1-y[i])) /\
                                    (s1 * pow(pi, y[i]) * pow(1-pi, 1-y[i]) + s2 * pow(p, y[i]) * pow(1-p, 1-y[i])
                                     + (1-s1-s2) * pow(q,y[i])*pow(1-q,1-y[i]))

    def M_step(self, y, n):
        """
        更新要求解的参数
        """
        mu = self.params['mu']
        mu2 = self.params['mu2']

        self.params['s1'][0] = sum(mu) / n
        self.params['s2'][0] = sum(mu2) / n
        self.params['pi'][0] = sum([mu[i] * y[i] for i in range(n)]) / sum(mu)
        self.params['p'][0] = sum([mu2[i] * y[i] for i in range(n)]) / sum(mu2)
        self.params['q'][0] = sum([(1-mu[i]-mu2[i]) * y[i] for i in range(n)]) / \
                              sum([1-mu_i-mu2_i for mu_i,mu2_i in zip(mu,mu2)])

    def fit(self, y):
        """
        进行模型求解
         y: 观测样本

        """
        n = len(y)
        self.__init_params(n)
        print(0, self.params['s1'], self.params['s2'], self.params['pi'], self.params['p'], self.params['q'])
        flag_begin = self.params['s1'][0] * self.params['pi'][0] + self.params['s2'][0] * self.params['p'][0] + (
                    1 - self.params['s1'][0] - self.params['s2'][0]) * self.params['q'][0]

        self.ele['s1'][0] = self.params['s1'][0]
        self.ele['s2'][0] = self.params['s2'][0]
        self.ele['pi'][0] = self.params['pi'][0]
        self.ele['p'][0] = self.params['p'][0]
        self.ele['q'][0] = self.params['q'][0]

        print(f'begin ---{flag_begin}')
        for i in range(self.n_epoch):
            self.E_step(y, n)
            self.M_step(y, n)
            print(i+1, self.params['s1'], self.params['s2'], self.params['pi'], self.params['p'], self.params['q'])
            self.ele['s1'][i+1] = self.params['s1'][0]
            self.ele['s2'][i+1] = self.params['s2'][0]
            self.ele['pi'][i+1] = self.params['pi'][0]
            self.ele['p'][i+1] = self.params['p'][0]
            self.ele['q'][i+1] = self.params['q'][0]

    def get_relusts(self, n, s1, s2, pi, p, q):
        #产生n个抛硬币的结果
        y = []
        for i in range(n):
            flag = np.random.rand(1)
            if flag < s1:
                if np.random.rand(1) < pi:
                    y.append(1)
                else:
                    y.append(0)
            if flag > s1 and flag < s2 +s1:
                if np.random.rand(1) < p:
                    y.append(1)
                else:
                    y.append(0)
            else:
                if np.random.rand(1) < q:
                    y.append(1)
                else:
                    y.append(0)
        return y

def run_three_coins_model():
    tcm = ThreeCoinsMode()
    s1, s2, pi, p, q = 0.7, 0.7, 0.7, 0.7, 0.8
    y = tcm.get_relusts(100, s1, s2, pi, p, q)
    tcm.fit(y)
    flag1 = s1 * pi + s2 * p + (1-s1-s2) * q
    flag2 = tcm.params['s1'][0]*tcm.params['pi'][0] + tcm.params['s2'][0]*tcm.params['p'][0] + (1 - tcm.params['s1'][0] - tcm.params['s2'][0])*tcm.params['q'][0]
    print(f'flag1 {flag1}\n')
    print(f'flag2 {flag2}')

#下面全是画图的内容
    x_label = list(range(1, tcm.n_epoch+2))
    fig, ax = plt.subplots()
    ax.plot(x_label, tcm.ele['s1'])
    ax.set_title('s1')
    ax.set_xlabel('iter')
    ax.set_ylabel('s1_value')

    fig, ax = plt.subplots()
    ax.plot(x_label, tcm.ele['s2'])
    ax.set_title('s2')
    ax.set_xlabel('iter')
    ax.set_ylabel('s2_value')

    fig, ax = plt.subplots()
    ax.plot(x_label, tcm.ele['pi'])
    ax.set_title('pi')
    ax.set_xlabel('iter')
    ax.set_ylabel('pi_value')

    fig, ax = plt.subplots()
    ax.plot(x_label, tcm.ele['p'])
    ax.set_title('p')
    ax.set_xlabel('iter')
    ax.set_ylabel('p_value')

    fig, ax = plt.subplots()
    ax.plot(x_label, tcm.ele['q'])
    ax.set_title('q')
    ax.set_xlabel('iter')
    ax.set_ylabel('q_value')

    plt.show()




if __name__ == '__main__':
    run_three_coins_model()




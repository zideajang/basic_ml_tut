## 极大后验概率估计(MAP 估计)
我们在极大似估计中，更加注重实际，也就是根据我们实验样本结果来进行估计，而忽视前人的经验，也就是我们只注重这次实验的结果。这样做在现实生活中显然是有问题，我们只看这个人近期表现呢，而没有看其过往做出的共享。所以才引入极大后验概率估计，其就是在极大似然的基础考虑先验的知识（也就是前人经验)。

$$\hat{\theta}_{MAP} = \arg \max_{\theta} P(\theta|X) = \arg \max_{\theta} \prod_{i=1}^N p(\theta) p(x_i|\theta) $$

$$\hat{\theta}_{MAP} = \arg \max_{\theta} \ln P(\theta|X) = \arg \max_{\theta} \left[  \ln p(\theta)  + \sum_{i=1}^N p(\theta) p(x_i|\theta) \right] $$




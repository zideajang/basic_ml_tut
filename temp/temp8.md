#### 似然函数

现在我想大家脑海里已经对贝叶斯定理有了印象。我们知道似然函数就是在有了证据前提下 先验概率。今天我们以似然函数为主来进行分析。

Bernoulli Distribution
Our example is that of a sequence of coin flips. We are interested in the probability of the coin coming up heads. In particular, we are interested in the probability of the coin coming up heads as a function of the underlying fairness parameter θ.

This will take a functional form, f. If we denote by k the random variable that describes the result of the coin toss, which is drawn from the set {1,0}, where k=1 represents a head and k=0 represents a tail, then the probability of seeing a head, with a particular fairness of the coin, is given by:
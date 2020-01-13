#### 似然函数

现在我想大家脑海里已经对贝叶斯定理有了印象。我们知道似然函数就是在有了证据前提下 先验概率。今天我们以似然函数为主来进行分析。

#### 伯努利分布
这是连续投币的实例
Our example is that of a sequence of coin flips. We are interested in the probability of the coin coming up heads. In particular, we are interested in the probability of the coin coming up heads as a function of the underlying fairness parameter θ.

This will take a functional form, f. If we denote by k the random variable that describes the result of the coin toss, which is drawn from the set {1,0}, where k=1 represents a head and k=0 represents a tail, then the probability of seeing a head, with a particular fairness of the coin, is given by:

$$P(k=1|\theta) = f(\theta)$$
$$P(k=1|\theta) = \theta$$
$$P(k=0|\theta) = 1 - \theta$$
$$p(k|\theta) = (1-\theta)^{(1 - k)}\theta^k$$

$$p({k_1,\dots k_N}|\theta) = \prod_{i=1} p(k_i|\theta)$$
$$p({k_1,\dots k_N}|\theta) = \prod_{i=1} \theta^{k_i}(1 - \theta)^{(1 - k_i)}$$

An extremely important step in the Bayesian approach is to determine our prior beliefs and then find a means of quantifying them.

In the Bayesian approach we need to determine our prior beliefs on parameters and then find a probability distribution that quantifies these beliefs.
In this instance we are interested in our prior beliefs on the fairness of the coin. That is, we wish to quantify our uncertainty in how biased the coin is.

To do this we need to understand the range of values that θ can take and how likely we think each of those values are to occur.

θ=0 indicates a coin that always comes up tails, while θ=1 implies a coin that always comes up heads. A fair coin is denoted by θ=0.5. Hence θ∈[0,1]. This implies that our probability distribution must also exist on the interval [0,1].

The question then becomes - which probability distribution do we use to quantify our beliefs about the coin?

Beta 分布
在这个实例中我们将选择 beta 分布 beta 分布的概率密度函数就是

P(θ|α,β)=θα−1(1−θ)β−1/B(α,β)

Where the term in the denominator, B(α,β) is present to act as a normalising constant so that the area under the PDF actually sums to 1.

I've plotted a few separate realisations of the beta distribution for various parameters α and β below:
```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    sns.set_palette("deep", desat=.6)
    sns.set_context(rc={"figure.figsize": (8, 4)})
    x = np.linspace(0, 1, 100)
    params = [
        (0.5, 0.5),
        (1, 1),
        (4, 3),
        (2, 5),
        (6, 6)
    ]
    for p in params:
        y = beta.pdf(x, p[0], p[1])
        plt.plot(x, y, label="$\\alpha=%s$, $\\beta=%s$" % p)
    plt.xlabel("$\\theta$, Fairness")
    plt.ylabel("Density")
    plt.legend(title="Parameters")
    plt.show()
```

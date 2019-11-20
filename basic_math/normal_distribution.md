### 概率密度函数
我们学习机器过程中，多次提到概率密度函数，那么什么是概率密度函数，我们怎么理解概率密度函数呢?我们假设一下有一个$x \in [0,1)$ 的数据，那么取到某一个数概率是多少呢，其实我们知道在$[0,1)$的数是无穷多，如果取到某一个数是均等的，无穷多个数概率 1 除以一个无穷大那么分配到取到每一个数概率就是无穷小也就是近视为 0，可以就是 0。那么我们再想想这些无穷小是怎么加和为 1。
我们将概率密度和密度对比来看，就好理解了，一个质量为 1 球上每一个点的质量也是近视为 0，我们通过密度而非质量来描述一个点，所以我们在0-1区间上所有数我们也是通过概率密度函数来表示。

对于随机变量 X 而言，其概率密度
$$ PDF:f_x(x) = \lim_{\Delta x \rightarrow 0} \frac{P(x \le X x + \Delta x)}{\Delta x} $$
概率密度函数就是数值

$$ =  \lim_{\Delta x \rightarrow 0} \frac{F_x(x+ \Delta x) - F_x(x)}{\Delta x} = F_x \prime(x)$$
CDF 进行微分就可以得到 PDF
$$ CDF F_x(x) \rightarrow PDF f_x(X)$$
而PDF 进行积分就可以得到 CDF
### 概率密度函数(PDF) 与概率之间的关系
$$P(a < X \le b) = F_x(b) - F_x(a)$$
$$ = \int_{-\infty}^b f_x(x)dx -  \int_{-\infty}^a f_x(x)dx $$
$$ = \int_a^b f_x(x)dx  $$
如果给你概率密度函数，然后我们就可以回答a 到 b 的概率。

$$ PDF:f_x(x) = \lim_{\Delta x \rightarrow 0} \frac{P(x \le X x + \Delta x)}{\Delta x} $$
当 $\Delta x $趋近无穷小时候
$$ P(x \le X \le x + \Delta x) \approx f_x(x) \cdot \Delta x$$

###概率密度函数(PDF)
$$ f_x(x) = F_x \prime (x)$$
$$ F_x(x) = \int_{- \infty}^x f_x(u)du$$
$$ P(a \le X \le b) = \int_a^b f_x(x)dx $$
$$ f_{-\infty}^{\infty} f_x(x)dx = 1 $$
$$f_x(x) \ge 0$$

**正态分布**(normal distribution)又名**高斯分布**(Gaussian distribution)，是一个非常常见的连续概率分布。正态分布在统计学上十分重要，经常用在自然和社会科学来代表一个不明的随机变量。

**正态分布**的数学期望值或期望值$\mu$ 等于位置参数，决定了分布的位置；其方差$\sigma^2$ 的开平方或标准差\sigma等于尺度参数，决定了分布的幅度。

其实我们做的工作多数根据经验而进行预测，经验就是我们通过采集数据来的,我们来想想一个问题，就是给我们 x 我们来估计其对应 y 是多少。

在数学中，连续型随机变量的**概率密度函数**(在不至于混淆时可以简称为密度函数)是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数。而随机变量的取值落在某个区域之内的概率则为概率密度函数在这个区域上的积分。当概率密度函数存在的时候，累积分布函数是概率 .

### 均匀分布
假设 $ X ~ U(a,b)$ 其概率密度为
$$ f(x) = \begin{cases}
    \frac{1}{b - a}, & a < x < b \\
    0
\end{cases} $$
则有期望 $E(X) = \int_{-\infty}^{\infty} x f(x) dx = \int_a^b \frac{1}{b-a} x dx = \frac{1}{2}(a+b)$
那么方差为 $D(X) = E(X^2) - [E(X)]^2$
$$ = \int_a^b x^2 \frac{1}{b-a} dx - \left( \frac{a+b}{2} \right) = \frac{(b-a)^2}{12}$$

### 指数分布
$$y = \lambda e^{-\lambda x}$$
假设随机变量 X 服从指数分布，其概率密度为
$$ f(x) = \begin{cases}
    \frac{1}{\theta} e^{-\frac{x}{\theta}} & x > 0 & \theta >0 \\
    0 & x \le 0
\end{cases} $$
则有
$$E(X) = \int_{-\infty}^{+\infty} x f(x) dx = \infty_0^{+\infty} x \cdot \frac{1}{\theta} e^{-\frac{x}{\theta}}dx$$
$$ - xe^{-\frac{x}{\theta}}|_0^{+\infty} + \int_{0}^{+\infty}e^{-\frac{x}{\theta}}dx = \theta$$

$$D(X) = E(X^2) - [E(X)]^2 = \infty_0^{+\infty} x^2 \cdot \frac{1}{\theta} e^{-\frac{x}{\theta}}dx - \theta^2 = 2\theta - \theta^2 = \theta^2$$
指数函数的一个重要特征是无记忆性(遗失记忆性)
对于时间，单一个人等车的时间，再去
$$P(X>s + t|x>s) = P(x>t)$$
如果 X 是我们家里买的吸顶灯，即使用了 s 小时，则供使用至少 s + t 小时的条件概率，与未使用开始至少使用 t 小时的概率相等。

### 正态分布
假X~ $N(\mu,\sigma^2)$,其概率密度为

$$ f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{\frac{-(x - \mu)^2}{2 \sigma^2}}  \sigma > 0 - \infty < x < + \infty $$
### Beta 分布
概率本身可不可以做研究，

$$P(X|\theta) \theta^z(1 - \theta)^{N-z}$$
$$ z = \sum_{i=1}^N X_i $$
$$ Beta(a,b) = \frac{\theta^{a - 1}(1 - \theta)^{b-1}}{B(a,b)}  $$

$$ P(\theta|data) = \frac{P(data|\theta)P(\theta)}{P(data)} $$
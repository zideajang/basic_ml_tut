$$l(x) = \log \prod_i \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x_i - \mu)^2}{2 \sigma^2}}$$
$$\begin{aligned}
    = \sum_i \log \frac{1}{\sqrt{2 \pi} \sigma} e^{- \frac{(x_i - \mu)^2}{2 \sigma^2}} \\
    = (\sum_i \log \frac{1}{\sqrt{2 \pi} \sigma}) + (\sum_i - \frac{(x_i - \mu)^2}{2 \sigma^2}) \\
    = \frac{n}{2} \log (2 \pi \sigma^2) - \frac{1}{2\sigma^2} \sum_i(x_i - \mu)^2
\end{aligned}$$

$$P(c|x) = \frac{p(x|c)p(c)}{p(x)}$$
这里 c 表示类别而 x 表示样板，这里假设 $c = \{c_1,c_2\}$ 也就是 c 有两种类别

$$p(c_1|x) + p(c_2|x) = 1$$
那么在给定 x 时候 x 属于 c1 和 c2 的概率相加和为 1

$$\begin{aligned}
    = \frac{p(x|c)p(c)}{p(x)} \\
    = \frac{p(x|c)p(c)}{\sum_jp(x|c_j)p(c_j)}
\end{aligned}$$ 

$$p(x|c) = p(x_1,x_2,\dots, x_n |c)$$
因为在朴素贝叶斯中假设特征间两两相对独立所以有
$$p(x_1,x_2,\dots,x_n) = p(x_1)p(x_2) \dots p(x_n)$$
$$p(x_1,x_2,\dots,x_n|c) = p(x_1|c)p(x_2|c) \dots p(x_n|c)$$

$$ p(x|c) = p(x_1,x_2,\dots,x_n|c) = p(x_1|c)p(x_2|c) \dots p(x_n|c)$$
$$p(x|c) = \frac{\prod_{i=1}^n p(x_i|c)p(c)}{\sum_jp(x|c_j)p(c_j)} $$

$$p(x|c) \propto \prod_{i=1}^n p(x_i|c) \underbrace{p(c)}_{prior}$$


今天是 2020 年第一天，小时候感觉这是充满想象和科幻的年代，当年也会根据自己从画报和杂志上科幻小说了解到对 2020 进行自己设定，不过今天我们不记得那时候我们对 2020 的憧憬。
回顾 2019 年，在简书坚持每天发表一篇文章，当然也是漏发的情况，但是还是合理利用规则坚持下来了。

其实文章涉及范围比较广，涉及 web、后端语言、以及一些底层语言例如 c++ 和 go。不过在步入下半年自己把工作之余时间都花费在机器学习上。起初只是赶潮流，看看弄弄，的确挺难几乎放弃，不过随着逐步深入和自己对其浓厚兴趣还是坚持到了今天。

从起初对于概念似懂非懂，到今天因为一个知识点我会翻查许多资料对其深究原理。因此又捡起了高等数学和线性代数。希望自己 2020 年能够发一些高质量的文章来回报您们对我鼓励和信任。

### 主题模型
主题模型的直观理解
pLSA模型和优化思路
Jensen 不等式以及变分EM
LDA 模型以及优化思路
主题模型应用

主题模型:给定很多文档，自动

模型假设:整个语料共享 K 个主题，每一个文档是 K 个主题的某一个未知比例的混合，每一个词属于某一个未知主题
- 中心思想
我们数据是若干篇文档，也称为语料。例如 1000篇 20 个主题
我们将文档通过 20 维度向量表示该文档与这些主题相关度
### 主题分布
主题模型做法就相当于降维，我们将一个若干词压缩为 20 维的向量。可以看做为聚类，soft 聚类。所以主题模型属于无监督学习模型。
- 先验分布
- 共轭分布
- Beta 分布
- Dirichlet 分布
- 三层贝叶斯网络模型 LDA
- Gibbs 采样和更新规律

LDA 的应用方向
- 信息提取和搜索
    - 语义分析
- 文档分类聚类、文章摘要、社区挖掘
- 基于内容的图像聚类、目标识别
- 生物信息数据的应用

虽然朴素贝叶斯的无法做语义分析因为无法解决一词多义和多词一义

#### gamma 函数
$$\int x^tdx$$
$$\int  e^xdx = e^{-x}$$
$$ \int x^t e^{-x} dx = f(t)$$
$$ \int t^{x} e^{-t} dt = f(x) $$

$$\Gamma(x) = \int_{0}^{+\infty} t^{x-1}e^{-t} = (x-1)!$$


alice figure what how likely sunny or likely rainny
sunny 10 2/3
rainny 5 1/3

transition 
s = 0.8s + 0.4r
r = 0.6r + 0.2s
s + r = 1

s = 2/3
r = 1/3

happy = 4/1 = s/r
grummpy = 2/3 = s/r

bayse theorem


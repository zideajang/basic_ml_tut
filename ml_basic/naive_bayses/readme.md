### Naive Bayes

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

$$P(y|X) = \frac{P(X|y)P(y)}{P(X)}$$

$$X = {x_1,x_2, \dots , x_n}$$

$$P(y|X) = \frac{P(x_1|y)P(y) \cdot P(x_2|y)P(y) \cdots P(x_n|y)P(y) }{P(X)}$$


$$ y = \arg \max_y P(y|X) = \arg \max_y \frac{P(x_1|y)P(y) \cdot P(x_2|y)P(y) \cdots P(x_n|y)P(y) }{P(X)}$$

$$ y = \arg \max_y \log(P(x_1|y)) +  \log(P(x_2|y)) + \dots +  \log(P(x_n|y)) + \log(P(y))$$

$$p(x_i|y) = \frac{}{} $$
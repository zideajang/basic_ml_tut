# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
# print X
y = [7,9,13,17.5,18]

model = LinearRegression()
model.fit(X,y)
test_pizza = np.array([[12]])
predicated_price = model.predict(test_pizza)[0]

print( "12 pizza should cost %.2f " % predicated_price)

# plt.figure()
# plt.title('Pizza Price')
# plt.xlabel('Diameter in inches')
# plt.ylabel('Price in dollors')
# plt.plot(X,y,'k.')
# plt.grid(True)
# plt.show()


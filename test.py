from enhance import *

n = 25
m = 25
a = [[0] * n for m in range(n)]
for i in range(n):
    for j in range(m):
        a[i][j] = j

a = np.array(a)
b = im2col(a, (5, 5))
c = col2mat(b, (5, 5))
print(b)
print(c)

from sympy import Matrix, pprint
import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt


def access_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bit_to_byte(data):
    pass


with open("D:\\Licenta\\test_image_short.png", "rb") as image:
    f = image.read()
    imgbytes = bytearray(f)
    print(imgbytes)
    bits = [access_bit(imgbytes, i) for i in range(len(imgbytes) * 8)]
    print(bits)
    print(len(imgbytes))
    str_bits = ""
    for b in bits:
        str_bits += str(b)
    print(str_bits)
    print(len(str_bits))

# q = 97
# x = np.random.randint(low=0, high=q - 1, size=(5, 10), dtype=int)
# pprint(Matrix(x) % q)
# print("")
# pprint(Matrix(x).echelon_form() % q)
# q = scipy.linalg.orth(x)
# pprint(Matrix(q))
# print(Matrix(q[:x.shape[1]]))
x = np.linspace(0, 10, 1000)
mu = 1
s = 1
pdf = np.exp((-np.pi * x ** 2) / s ** 2)
cdf = np.cumsum(pdf)
cdf = cdf / cdf[-1]
plt.figure()
plt.subplot(121)
plt.plot(x, pdf)
plt.subplot(122)
plt.plot(x, cdf)
uniform_samples = np.random.uniform(0, 1, 100000)
index = []
for sample in uniform_samples:
    index.append(np.argmin(np.abs(cdf - sample)))
pdf_samples = x[index]
plt.hist(pdf_samples, bins=100)
plt.show()
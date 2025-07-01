import numpy as np
import matplotlib.pyplot as plt
filter1 = np.random.randn(100,3,3,2) # this shows 100 filters of dimension 3x3x2 i.e a 3x3 grid with 2 channels
x = np.random.randn(4, 3, 3, 2) # same way the input image can be described, but care must be taken as the no of channels of both the i/p img anf the filter must be same
b = np.zeros((100 , 1 , 1 , 1))


def zero_pad(x , pad):
    x_pad = np.pad(x , ((0,0) , (pad,pad) , (pad,pad) , (0,0)) ,mode = 'constant' , constant_values = (0,0))
    return x_pad


x_pad = zero_pad(x, 3)
print(x.shape)
print(x_pad.shape)

def basic_conv(prev_slice , W , b):
    s = np.multiply(prev_slice , W)
    Z = np.sum(s)
    Z += float(b)
    return Z

def first_conv(x_pad , filter1 ,b,stride = 2):
    m, l1 , b1 , _ = x_pad.shape
    f_count, l2 , b2 , _ = filter1.shape
    l_f = int((l1 - l2) / stride) + 1
    b_f = int((b1 - b2) / stride) + 1
    Z = np.zeros((m , l_f , b_f , f_count))

    for i in range(m):
        for j in range(l_f):
            for k in range(b_f):
                for l in range(f_count):
                    vert_initial = stride * j
                    vert_end = vert_initial + l2
                    horizontal_start = stride * k
                    horizontal_end = horizontal_start + b2

                    a_slice = x_pad[i, vert_initial:vert_end, horizontal_start:horizontal_end, :]
                    W = filter1[l]
                    Z[i, j, k, l] = basic_conv(a_slice, W, b[l])
    return Z   

output1 = first_conv(x_pad , filter1 , b , stride = 2)
print(output1.shape)

def max_pool(Z , size = 2 , stride = 2):
    m , l1 , b1 , c = Z.shape
    l_f = int((l1 - size) / stride) + 1
    b_f = int((b1 - size) / stride) + 1
    Z_pooled = np.zeros((m , l_f , b_f , c))

    for i in range(m):
        for j in range(l_f):
            for k in range(b_f):
                for ch in range(c):
                    vert_initial = j * stride
                    vert_end = vert_initial + size
                    horizontal_start = k * stride
                    horizontal_end = horizontal_start + size

                    slice = Z[i, vert_initial:vert_end, horizontal_start:horizontal_end, ch]
                    Z_pooled[i, j, k, ch] = np.max(slice)
    return Z_pooled               

output2 = max_pool(output1 , size = 2 , stride = 2)
print(output2.shape)

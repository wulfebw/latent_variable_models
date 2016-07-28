
import numpy as np
import time


def babylonian_sqrt(a, max_iterations=1000, threshold=1e-5):
    prev_sqrt = sqrt = a
    for idx in range(max_iterations):
        sqrt = .5 * (a / sqrt + sqrt)
        if abs(prev_sqrt - sqrt) < threshold:
            break
    return sqrt

def time_function_for_values(function, values):
    start = time.time()
    for v in values:
        function(v)
    end = time.time()
    return end - start
    
if __name__ == '__main__':
    num_values = 1000
    a_values = np.arange(num_values) + 1
    btime = time_function_for_values(babylonian_sqrt, a_values)
    print 'babylonian took: {} seconds'.format(btime)
    nptime = time_function_for_values(np.sqrt, a_values)
    print 'numpy took: {} seconds'.format(nptime)
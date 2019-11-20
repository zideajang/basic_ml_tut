# coding=utf-8
import numpy as np

# smaple 10 integers
def demo_one():
    sample = np.random.randint(low=1,high=100,size=10)
    print('Original sample: %s' % sample)
    print('Sample mean: %s' % sample.mean())
    resamples = [np.random.choice(sample,size=sample.shape) for i in range(100)]
    resamples_means = np.array([resample.mean() for resample in resamples])
    print('Mean of re-samples\' means: %s ' % resamples_means.mean())

'''
Original sample: [35 57  4 99 43 43 55 21 36 42]
Sample mean: 43.5

'''

if __name__ == "__main__":
    demo_one()
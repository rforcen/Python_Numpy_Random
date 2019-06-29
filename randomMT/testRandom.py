import numpy
from RandomMT import RandomMT as rmt
from timeit import default_timer as timer
from cmath import sqrt

def compRandom():
    print('random timing test...')
    size = 1024 * 1024 * 32
    tn = timer()
    rndArr = numpy.random.random(size)
    tn = timer() - tn
    print('time numpy for ', size, 'values', tn)

    tr = timer()
    rndArr = rmt.rand(size, 'f8')
    tr = timer() - tr
    print('time RandomMT for ', size, 'values', tr, 'ratio numpy/mt:', tn/tr)

    rndArr = rmt.rand(size, 'f8')
    print('check rand-f8,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 1))
    rndArr = rmt.rand(size, 'f4')
    print('check rand-f4,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 1))
    rndArr = rmt.rand(size, 'i1')
    print('check rand-i1,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 2 ** 7 - 1))
    rndArr = rmt.rand(size, 'i2')
    print('check rand-i2,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 2 ** 15 - 1))
    rndArr = rmt.rand(size, 'u1')
    print('check rand-u1,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 2 ** 8 - 1))
    rndArr = rmt.rand(size, 'u2')
    print('check rand-u2,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 2 ** 16 - 1))
    rndArr = rmt.rand(size, 'i4')
    print('check rand-i4,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 2 ** 31 - 1))
    rndArr = rmt.rand(size, 'i8')
    print('check rand-i8,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 2**64-1))
    rndArr = rmt.rand(size, 'u4')
    print('check rand-u4,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 2 ** 32 - 1))
    rndArr = rmt.rand(size, 'u8')
    print('check rand-u8,', numpy.all(rndArr >= 0) and numpy.all(rndArr <= 2 ** 64 - 1))

    rndArr = rmt.rand(size, 'c4')
    print('check rand-c4,', numpy.all(numpy.abs(rndArr) >= 0) and numpy.all(numpy.abs(rndArr) <= sqrt(2)))
    rndArr = rmt.rand(size, 'c8')
    print('check rand-c8,', numpy.all(numpy.abs(rndArr) >= 0) and numpy.all(numpy.abs(rndArr) <= sqrt(2)))

compRandom()
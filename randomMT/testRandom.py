'''
test RandomMT with type check and timing performance
'''
import numpy
from RandomMT import RandomMT as rmt
from timeit import default_timer as timer


def compRandom():
    def testTypes():
        def check(type, range):
            rndArr = rmt.rand(size, type)
            print('check rand -', type, '->',
                  'ok' if numpy.all(rndArr >= range[0]) and numpy.all(rndArr <= range[1]) else 'fail')

        for c in [['f8', [0, 1]], ['f4', [0, 1]], ['i1', [0, 2 ** 7 - 1]], ['i2', [0, 2 ** 15 - 1]],
                  ['u1', [0, 2 ** 8 - 1]], ['u2', [0, 2 ** 16 - 1]], ['i4', [0, 2 ** 31 - 1]], ['i8', [0, 2 ** 64 - 1]],
                  ['u4', [0, 2 ** 32 - 1]], ['u8', [0, 2 ** 64 - 1]],
                  ['c4', [0, complex(1, 1)]], ['c8', [0, complex(1, 1)]]]:
            check(c[0], c[1])

    def testTiming():
        print('random timing test...')

        tn = timer()
        rndArr = numpy.random.random(size)
        tn = timer() - tn
        print('time numpy for ', size, 'values', tn)

        tr = timer()
        rndArr = rmt.rand(size, 'f8')
        tr = timer() - tr
        print('time RandomMT for ', size, 'values', tr, 'ratio numpy/mt:', tn / tr)

    size = 50000000

    testTiming()
    testTypes()


compRandom()

/*
    multithraded numpy random generator
    runs 6 x times faster than numpy.random.random(size)
    supports types: i1, u1, i2, u2, i4, u4, i8, u8, f4, f8

    usage:
    from RandomMT import RandomMT as rmt
    ru = rmd.rand(1000, 'i2') # generate 1000 uint16 randoms
*/

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <string.h>
#include <complex>

#include "Thread.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

using std::is_same;
using std::complex;

class RandomMT {
    typedef np::ndarray (*RandFunc) (int n);

public:
     RandomMT() {
        init();
     }

     static np::ndarray rand(int n, char* type) {
        const char *types="i1i2u1u2u4u8i4i8f4f8c4c8";

        const RandFunc fv[]=  // funcs matching 'types' array
        {
            rand_1_2<int8_t>,   rand_1_2<int16_t>,  rand_1_2<uint8_t>, rand_1_2<uint16_t>,
            rand_4_8<uint32_t>, rand_4_8<uint64_t>, rand_4_8<int32_t>, rand_4_8<int64_t>,
            rand_4_8<float>,    rand_4_8<double>,
            rand_complex<float>, rand_complex<double>
        };
        auto p=strstr(types, type);
        if (p) return fv[(p-types)/2](n);

        return rand_4_8<double>(n); // default
     }

private:
     static bool checkType(const char*type, const char*tp) {
         return !strncmp(type, tp, 2);
     }

     template<typename T>
     static np::ndarray rand_4_8(int n) { // 4, 8 bytes int & float
        T*rv=new T[n]; // random vector

        T div = ((T)0.1==0) ? 1 : RAND_MAX; // is int?

        Thread(n).run([&rv, div](int t, int from, int to) {
            unsigned seed;
            for (int i=from; i<to; i++)
                rv[i]=(T)::rand_r(&seed) / div;
        });

        return np::from_data(rv,   // data -> rv
                np::dtype::get_builtin<T>(),      // dtype -> T
                p::make_tuple(n), // shape -> n
                p::make_tuple(1*sizeof(T)), p::object());     // stride 1
     }

     template<typename T>
     static np::ndarray rand_1_2(int n) { // 1, 2 bytes int
        T*rv=new T[n];

        T mod; // calc modulus to fit random values
        if (is_same<T, uint8_t>::value)     mod=0xffu;
        if (is_same<T, uint16_t>::value)    mod=0xffffu;
        if (is_same<T, int8_t>::value)      mod=0x7f;
        if (is_same<T, int16_t>::value)     mod=0x7fff;

        Thread(n).run([&rv, mod](int t, int from, int to) {
            unsigned seed;
            for (int i=from; i<to; i++)
                rv[i]=(T) (::rand_r(&seed) % mod);
        });

        return np::from_data(rv,   // data -> rv
                np::dtype::get_builtin<T>(),      // dtype -> T
                p::make_tuple(n), // shape -> n
                p::make_tuple(1*sizeof(T)), p::object());     // stride 1
     }

     template<typename T>
     static np::ndarray rand_complex(int n) { // 4, 8 bytes complex
        complex<T>*rv=new complex<T>[n]; // random vector

        Thread(n).run([&rv](int t, int from, int to) {
            unsigned seed;
            for (int i=from; i<to; i++)
                rv[i]=complex<T>((T)::rand_r(&seed) / RAND_MAX, (T)::rand_r(&seed) / RAND_MAX);
        });

        return np::from_data(rv,   // data -> rv
                np::dtype::get_builtin<complex<T>>(),      // dtype -> T
                p::make_tuple(n), // shape -> n
                p::make_tuple(1*sizeof(complex<T>)), p::object());     // stride 1
     }

     static unsigned init() {
        Py_Initialize(); // init boost & numpy boost
        np::initialize();
        return 0;
     }

     static unsigned int junk;
};

unsigned RandomMT::junk=RandomMT::init(); // make sure init is executed

BOOST_PYTHON_MODULE(RandomMT) {
    p::class_<RandomMT>("RandomMT", p::init<>())
        .def("rand", &RandomMT::rand);
}


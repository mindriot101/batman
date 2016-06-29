import pytest
import batman
import numpy as np
from functools import partial
from hypothesis import given
import hypothesis.strategies as st
from cffi import FFI

ffi = FFI()
ffi.cdef(
	"""
	typedef struct {
		double c1;
		double c2;
		double c3;
		double c4;
	} NonlinearLimbDarkeningParameters;

	typedef struct {
	double t0;
	double per;
	double rp;
	double a;
	double inc;
	double ecc;
	double w;
	union {
		NonlinearLimbDarkeningParameters ldc;
	};
	} Params;

	double *light_curve(Params *params, double *t, int length);
	"""
)

C = ffi.dlopen('libbatman.so')

@pytest.fixture
def hjd():
    return np.linspace(-0.5, 0.5, 1000)

@pytest.fixture
def period():
    return 1.0

@pytest.fixture
def t0():
    return 0.

@pytest.fixture
def a():
    return 15.

# @pytest.fixture
# def rp():
#     return 0.1

@pytest.fixture
def inc():
    return 87.

@pytest.fixture
def c1():
    return 0.7692

@pytest.fixture
def c2():
    return -0.716

@pytest.fixture
def c3():
    return 1.1874

@pytest.fixture
def c4():
    return 0.5372

@pytest.fixture
def ecc():
    return 0.

@pytest.fixture
def w():
    return 0.

nice_floats = partial(st.floats, allow_nan=False, allow_infinity=False)

@given(
    rp=nice_floats(min_value=0.01, max_value=0.1),
)
def test_stuff(hjd, period, t0, rp, a, inc, ecc, w, c1, c2, c3, c4):
    params = batman.TransitParams()
    params.per = period
    params.t0 = t0
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.limb_dark = 'nonlinear'
    params.u = [c1, c2, c3, c4]

    model = batman.TransitModel(params, hjd)
    python_lc = model.light_curve(params)

    params = ffi.new('Params *')
    params.per = period
    params.t0 = t0
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.ldc = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4}

    phjd = ffi.cast('double *', hjd.ctypes.data)
    pflux = C.light_curve(params, phjd, hjd.size)
    c_lc = np.frombuffer(ffi.buffer(pflux, hjd.size * 8), dtype=np.float64)

    assert np.allclose(python_lc, c_lc)

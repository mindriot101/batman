/* The batman package: fast computation of exoplanet transit light curves
 * Copyright (C) 2015 Laura Kreidberg	 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef BATMAN_PYTHON
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"
#endif

#include <math.h>

#if defined (_OPENMP)
#  include <omp.h>
#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

void eclipse(double *ds, double *fs, int len, double p, double fp, int nthreads) {
  int i;
  double d, kap1, kap0, alpha_t, alpha_o;

  if (fabs(p - 0.5) < 1.e-3) {
    p = 0.5;
  }

#if defined (_OPENMP)
  omp_set_num_threads(nthreads);
#endif

#if defined (_OPENMP)
#pragma omp parallel for private(d, kap1, kap0)
#endif
  for(i=0; i<len; i++)
  {
    d = ds[i]; 						// separation of centers

    if(d >= 1. + p) {
      fs[i] = 1. + fp;					//planet fully visible
    } else if(d < 1. - p) {
      fs[i] = 1.;					//planet fully occulted
    } else {								//planet is crossing the limb
      kap1=acos(fmin((1. - p*p + d*d)/2./d, 1.));
      kap0=acos(fmin((p*p + d*d - 1.)/2./p/d, 1.));
      alpha_t = (p*p*kap0 + kap1 - 0.5*sqrt(fmax(4.*d*d \
	      - pow(1. + d*d - p*p, 2.), 0.)))/M_PI;		//transit depth
      alpha_o = alpha_t/p/p;				 	//fraction of planet disk that is eclipsed by the star
      fs[i] = 1. + fp*(1. - alpha_o);			//planet partially occulted
    }
  }
}

#ifdef BATMAN_PYTHON
static PyObject *_eclipse(PyObject *self, PyObject *args)
{
	int nthreads;
	double p, fp;

	PyArrayObject *ds, *flux;
	npy_intp dims[1];
	
  	if(!PyArg_ParseTuple(args, "Oddi", &ds, &p, &fp, &nthreads)) return NULL;		//parses function input

	dims[0] = PyArray_DIMS(ds)[0]; 
	flux = (PyArrayObject *) PyArray_SimpleNew(1, dims, PyArray_TYPE(ds));	//creates numpy array to store return flux values
	
	double *f_array = PyArray_DATA(flux);
	double *d_array = PyArray_DATA(ds);

	eclipse(d_array, f_array, dims[0], p, fp, nthreads);

	return PyArray_Return((PyArrayObject *)flux);
}


static char _eclipse_doc[] = "This extension module returns a limb darkened light curve for a uniform stellar intensity profile.";

static PyMethodDef _eclipse_methods[] = {
  {"_eclipse", _eclipse, METH_VARARGS, _eclipse_doc},{NULL}};

#if PY_MAJOR_VERSION >= 3
	static struct PyModuleDef _eclipse_module = {
		PyModuleDef_HEAD_INIT,
		"_eclipse",
		_eclipse_doc,
		-1, 
		_eclipse_methods
	};

	PyMODINIT_FUNC
	PyInit__eclipse(void)
	{
		PyObject* module = PyModule_Create(&_eclipse_module);
		if(!module)
		{
			return NULL;
		}
		import_array(); 
		return module;
	}
#else

	void init_eclipse(void)
	{
	  	Py_InitModule("_eclipse", _eclipse_methods);
		import_array(); 
	}
#endif
#endif // BATMAN_PYTHON


from distutils.core import setup, Extension

module = Extension('_C_camodule', ['_C_camodule.c'], 
      include_dirs = ["/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/"])
module.extra_compile_args = ['--std=c99']

setup(name='_C_camodule', version='1.0', ext_modules=[module])
      
#adres do arrayobject.h: /usr/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h

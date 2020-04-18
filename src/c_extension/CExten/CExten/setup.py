from distutils.core import setup, Extension

MOD = 'UCB' #模块名
setup(name=MOD,ext_modules=[Extension(MOD,sources=['./CExten.cpp'])])
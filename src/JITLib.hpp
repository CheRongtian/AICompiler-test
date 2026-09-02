#ifndef JITLIB_HPP
#define JITLIB_HPP

#ifdef _WIN32
#define DLLEXPORT _declspec(dellexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT double putchard(double X);

extern "C" DLLEXPORT double printd(double X);

#endif
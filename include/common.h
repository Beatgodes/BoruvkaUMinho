#ifndef COMMON
#define COMMON


#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <utility>
#include <sys/time.h>
#include <vector>
#include <string.h>


#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	return (stat (name.c_str(), &buffer) == 0);
}

static double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

#endif
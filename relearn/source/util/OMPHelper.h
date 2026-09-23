#pragma once

#ifdef _OPENMP
#include <omp.h>
#else
[[maybe_unused]] static void omp_set_num_threads([[maybe_unused]] int num) { }
[[maybe_unused]] static int omp_get_thread_num() { return 1; }
#endif
#include "pomp_lib.h"

struct ompregdescr omp_rd_5 = {
  "parallel", "", 0, "mult-sync.c", 25, 25, 35, 35
};

#define POMP_DLIST_00005 shared(omp_rd_5,omp_rd_6,omp_rd_7,omp_rd_8)

struct ompregdescr omp_rd_6 = {
  "barrier", "", 0, "mult-sync.c", 29, 29, 0, 0
};

struct ompregdescr omp_rd_7 = {
  "for", "", 0, "mult-sync.c", 30, 30, 32, 32
};

struct ompregdescr omp_rd_8 = {
  "single", "", 0, "mult-sync.c", 33, 33, 34, 34
};



class timer {
   double start_time;
   double time;
public:
   inline timer() {
      time = 0.0;
   }
   inline void stop() {
   	struct timespec res;
   	clock_gettime(CLOCK_MONOTONIC,  &res);
   	const double stop_time = res.tv_sec + res.tv_nsec / 1e9;
      time += stop_time - start_time;
   }
   inline void start() {
   	struct timespec res;
   	clock_gettime(CLOCK_MONOTONIC,  &res);
   	start_time = res.tv_sec + res.tv_nsec / 1e9;
   }
   inline void reset() {
      time = 0.0;
   }
   inline double read() {
      return time;
   }
};



template<class T>
inline vec3<T> distance(const vec3<T>& a, const vec3<T>& b) {
	vec3<T> d;
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		d[dim] = distance(a[dim], b[dim]);
	}
	return d;
}


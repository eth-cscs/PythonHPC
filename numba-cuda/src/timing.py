import numba.cuda as cuda
import time


class time_region:
    def __init__(self, time_offset=0):
        self._time_off = time_offset

    def __enter__(self):
        self._t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._t_end = time.time()

    def elapsed_time(self):
        return self._time_off + (self._t_end - self._t_start)


class time_region_cuda:
    def __init__(self, time_offset=0, cuda_stream=0):
        self._t_start = cuda.event(timing=True)
        self._t_end = cuda.event(timing=True)
        self._time_off = time_offset
        self._cuda_stream = cuda_stream

    def __enter__(self):
        self._t_start.record(self._cuda_stream)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._t_end.record(self._cuda_stream)
        self._t_end.synchronize()

    def elapsed_time(self):
        return self._time_off + 1.e-3*cuda.event_elapsed_time(self._t_start,
                                                              self._t_end)

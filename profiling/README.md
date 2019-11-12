# Using the line and memory profilers


## `line_profiler`
The `line_profiler` demonstration is inside the notebook `01_line_profiler.ipynb`.

## `memory_profiler`
Here we use `memory_profiler` by putting the code we want to profile in a function and decorate it with `memory_profiler.profile`.
Then, running it normally
```
pyhton 02_memprofiler.py
```
will print the memory usage of the function by line.

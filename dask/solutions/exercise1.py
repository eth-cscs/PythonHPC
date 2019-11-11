x = [dask.delayed(square)(i) for i in range(10)]
y = dask.delayed(sum)(x)

# use `y.visualize()` to visualize the graph and `y.compute()` to execute it.

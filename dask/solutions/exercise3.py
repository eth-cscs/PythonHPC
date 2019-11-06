x = []
for i in range(10):
    if i % 2 == 0:
        x.append(dask.delayed(square)(i))
    else:
        x.append(dask.delayed(add)(i, i))

y = dask.delayed(sum)(x)

# use `y.visualize()` to visualize the graph and `y.compute()` to execute it.

x = []
for i in range(10):
    x.append(dask.delayed(square)(i))

y = dask.delayed(sum)(x)

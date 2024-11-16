import timeit
import functools


def benchmark(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t = timeit.Timer(lambda: func(*args, **kwargs))
        duration = t.timeit(number=1)
        print(f"Benchmark: {func.__name__} took {duration:.6f} seconds")

    return wrapper

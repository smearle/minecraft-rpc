import math

def square_spiral(n: int):
    # Parameterize square spiral as on https://math.stackexchange.com/a/3158068
    flr_sqrt_n = math.floor(math.sqrt(n))
    n_hat = flr_sqrt_n if flr_sqrt_n % 2 == 0 else flr_sqrt_n - 1
    n_hat_sqr = n_hat ** 2
    if n_hat_sqr <= n <= n_hat_sqr + n_hat:
        x = - n_hat / 2 + n - n_hat_sqr
        y = n_hat / 2
    elif n_hat_sqr + n_hat < n <= n_hat_sqr + 2 * n_hat + 1:
        x = n_hat / 2
        y = n_hat / 2 - n + n_hat_sqr + n_hat
    elif n_hat_sqr + 2 * n_hat + 1 < n <= n_hat_sqr + 3 * n_hat + 2:
        x = n_hat / 2 - n + n_hat_sqr + 2 * n_hat + 1
        y = - n_hat / 2 - 1
    elif n_hat_sqr + 3 * n_hat + 2 < n <= n_hat_sqr + 4 * n_hat + 3:
        x = - n_hat / 2 - 1
        y = - n_hat / 2 - 1 + n - n_hat_sqr - 3 * n_hat - 2
    else: raise Exception
    return int(x), int(y)

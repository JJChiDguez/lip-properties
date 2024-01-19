#!/usr/bin/sage -python
# -*- coding: utf8 -*-

import sys
from timeit import timeit
from timeit import default_timer as timer
import multiprocessing as mp

from algorithms import (
	arguments,
	counter,
    group_action,
    recover,
    recover_gb,
	sampling_quadratic_form,
	sampling_from_dsq,
    sampling_unimodular_matrix,
    create_to_csv_file
)

from sage.all import (
    ZZ,
	e,
	log,
	pi,
	sqrt,
    identity_matrix
)

# @counter
def main(n: int):

    # Instance
    q, random_automorphism = sampling_quadratic_form(n, verbose=False)
    b, _ = q.cholesky().gram_schmidt()
    s = b.norm() * sqrt(log(2 * n + 4, e) / pi)
    assert (q.is_symmetric())
    assert (q == q.transpose())

    u, q_ = sampling_from_dsq(q, n, s)
    assert q_ == group_action(u, q)

    # Set to zero counters
    group_action.count = 0
    sampling_from_dsq.count = 0
    recover.count = 0
    sampling_unimodular_matrix.count = 0

    # Main calculations
    @counter
    def randomized_oracle():
        # We did not use group_action(w * u, function(v, q)) to isolate the oracle cost
        _, _q = sampling_from_dsq(q, n, s)
        return _q, u.transpose() * _q * u

    m = n * (n + 1) // 2
    
    # Sampling
    time0 = timer()
    q0 = []
    q0.append(q)
    q1 = []
    q1.append(q_)
    sys.stdout.write("\r# Sampling from the randomized oracle: {}/{}".format(1, m))
    sys.stdout.flush()
    for i in range(0, m - 1, 1):
        sys.stdout.write("\r# Sampling from the randomized oracle: {}/{}".format(i + 2, m))
        sys.stdout.flush()
        q0_i, q1_i = randomized_oracle()
        q0.append(q0_i)
        q1.append(q1_i)
    
    # Linearization
    time1 = timer()
    full_rank, recovered_u = recover(q0, q1, n)

    # Groebner Basis with n(n+1)/2 equations
    time2 = timer()
    full_rank, recovered_u = recover_gb(q0, q1, n)

    # # Groebner Basis with nlog2(n) equations
    time3 = timer()
    # m = n * (n.bit_length())
    # full_rank, recovered_u = recover_gb(q0[:m], q1[:m], n)

    # # Groebner Basis with nlog2(n)/2 equations
    # time4 = timer()
    # m = n * (n.bit_length()) // 2
    # full_rank, recovered_u = recover_gb(q0[:m], q1[:m], n)

    # time5 = timer()

    output = {
        'n': n,
        'sampling': time1 - time0,
        'Recover': time2 - time1,
        'RecoverGB with m=n(n+1)/2': time3 - time2,
        # 'RecoverGB with m=nlog2(n)': time4 - time3,
        # 'RecoverGB with m=nlog2(n)/2': time5 - time4,
    }

    return output


if __name__ == '__main__':
    # Setup
    n = arguments(sys.argv[1:]).dimension

    # main()
    tries = 25
    n_cores = mp.cpu_count()
    print(f'\n#(total cores): {n_cores}')
    with mp.Pool(n_cores // 2) as pool:
        print(f'#(used cores):  {pool._processes}\n')
        bench = pool.starmap(main, [(n,)] * tries)
    
    create_to_csv_file(f'Experiments_n={n}', bench)


    # print(f'\n{"#" * 27} It took {timeit(lambda: main(), number=tries)} seconds\n')
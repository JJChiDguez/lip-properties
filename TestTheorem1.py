#!/usr/bin/sage -python
# -*- coding: utf8 -*-

import sys
from timeit import timeit
from timeit import default_timer as timer

from algorithms import (
	arguments,
	counter,
    group_action,
    print_matrix,
    recover,
    recover_gb,
	sampling_quadratic_form,
	sampling_from_dsq,
    sampling_unimodular_matrix
)

from sage.all import (
    ZZ,
	e,
	log,
	pi,
	sqrt,
    identity_matrix
)

@counter
def main():
    # Setup
    n = arguments(sys.argv[1:]).dimension
    verbose = arguments(sys.argv[1:]).verbose
    groebner_basis = arguments(sys.argv[1:]).groebner_basis
    Recover = {True:recover_gb, False:recover}[groebner_basis]

    if main.count == 1:
        print(f'\n# n:     \t{n}')
        print(f'# Oracle:\tRandomized oracle as in Theorem 1')
        print(f'# Verbose:\t{verbose}')
        print(f'# Gröbner base:\t{groebner_basis}')

    # Instance
    q, random_automorphism = sampling_quadratic_form(n, verbose=verbose)
    b, _ = q.cholesky().gram_schmidt()
    s = b.norm() * sqrt(log(2 * n + 4, e) / pi)
    assert (q.is_symmetric())
    assert (q == q.transpose())

    u, q_ = sampling_from_dsq(q, n, s)
    assert q_ == group_action(u, q)

    # Set to zero counters
    group_action.count = 0
    sampling_from_dsq.count = 0
    Recover.count = 0
    sampling_unimodular_matrix.count = 0

    if verbose:
        print(f'{"#" * 27} Public parameters')
        print_matrix(q, 'q')
        print_matrix(q_, 'q_')
        print(f'\n{"#" * 27} Secret unimodular matrix')
        print_matrix(u, 'u')

    # Main calculations
    @counter
    def randomized_oracle():
        # We did not use group_action(w * u, function(v, q)) to isolate the oracle cost
        _, _q = sampling_from_dsq(q, n, s)
        return _q, u.transpose() * _q * u

    m = n * (n + 1) // 2
    v = identity_matrix(ZZ, n)
    recovered_u = None
    full_rank = False
    loop = 0
    print('')
    while not full_rank:
        # Sampling
        time0 = timer()
        loop += 1
        q0 = []
        q0.append(q)
        q1 = []
        q1.append(q_)
        sys.stdout.write("\r# [#{}] Sampling from the randomized oracle: {}/{}".format(loop, 1, m))
        sys.stdout.flush()
        for i in range(0, m - 1, 1):
            sys.stdout.write("\r# [#{}] Sampling from the randomized oracle: {}/{}".format(loop, i + 2, m))
            sys.stdout.flush()
            q0_i, q1_i = randomized_oracle()
            q0.append(q0_i)
            q1.append(q1_i)
        time1 = timer()
        # Solving the linear system
        time2 = timer()
        full_rank, recovered_u = Recover(q0, q1, n)
        time3 = timer()

    assert full_rank
    print(f'\n\n# Elapsed time (Oracle):\t{time1 - time0}')
    print(f'# Elapsed time (Recover):\t{time3 - time2}\n')

    # To recover UR
    r = identity_matrix(ZZ, n)
    while recovered_u is None:
        # Randomizing
        time0 = timer()
        r = sampling_unimodular_matrix(n)
        assert(r.det() ** 2 == 1)
        q1_ = [ group_action(r, q1_k) for q1_k in q1 ]
        time1 = timer()
        # Solving the linear system
        time2 = timer()
        full_rank, recovered_u = Recover(q0, q1_, n)
        time3 = timer()
        
        print(f'# Elapsed time (Randomize):\t{time1 - time0}')
        print(f'# Elapsed time  (Recover):\t{time3 - time2}')
        assert full_rank

    recovered_u = recovered_u * r.inverse()


    # Validate solution
    assert recovered_u.det() ** 2 == 1
    assert recovered_u == u or recovered_u == -u
    assert recovered_u.transpose() * q * recovered_u == q_

    # Print info
    if not verbose:
        print(f'\n{"#" * 27} iteration {main.count}')
    else:
        print(f'\n{"#" * 27} Recovered unimodular matrix')
        print_matrix(recovered_u, "recovered_u")
        print(f'\n{"#" * 27} Complexity\n')


    print(f'# {group_action.count} group action calls')
    print(f'# {randomized_oracle.count} oracle calls concerning U ({sampling_from_dsq.count} calls to Dₛ([Q]))')
    print(f'# {Recover.count} calls to Recover()')
    print(f'# {sampling_unimodular_matrix.count} calls to SampleUₜ()')


if __name__ == '__main__':
    # main()
    tries = {True: 1, False: 25}[arguments(sys.argv[1:]).verbose]
    print(f'\n{"#" * 27} It took {timeit(lambda: main(), number=tries)} seconds\n')
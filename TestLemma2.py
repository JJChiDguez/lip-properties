#!/usr/bin/sage -python
# -*- coding: utf8 -*-

import sys
from timeit import timeit

from algorithms import (
    algorithm,
    arguments,
    counter,
    group_action,
    print_matrix,
    matrix,
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
        print(f'# Oracle:\tTQFP oracle as in Lemma 2')
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
    def function(v: matrix, p: matrix):  # It computes: V×Q×Vᵀ
        return group_action(v.transpose(), p)

    # Decorate oracle_call() with the counter() decorator
    @counter
    def oracle_call(v: matrix):
        # It simulates: given (Vᵀ×U)×Q×(Uᵀ×V), returns Uᵀ×(V×Q×Vᵀ)×U
        w = random_automorphism().transpose()
        assert w.transpose() * q * w == q
        # We did not use group_action(w * u, function(v, q)) to isolate the oracle cost
        _q = v * q * v.transpose()
        return (w * u).transpose() * _q * (w * u)

    recovered_u = algorithm(q, n, s, function, (oracle_call,), Recover=Recover)

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


    print(f'# {group_action.count} group action calls ({function.count} of them concern with input Vᵀ)')
    print(f'# {oracle_call.count} oracle calls concerning U')
    print(f'# {sampling_from_dsq.count} calls to Dₛ([Q])')
    print(f'# {Recover.count} calls to Recover()')
    print(f'# {sampling_unimodular_matrix.count} calls to SampleUₜ()')


if __name__ == '__main__':
    # main()
    tries = {True: 1, False: 25}[arguments(sys.argv[1:]).verbose]
    print(f'\n{"#" * 27} It took {timeit(lambda: main(), number=tries)} seconds\n')
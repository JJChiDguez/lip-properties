#!/usr/bin/sage -python
# -*- coding: utf8 -*-

from timeit import default_timer as timer
import sys
import argparse
import csv

# SageMath imports
from sage.all import (
    deepcopy,
    e,
    floor,
    log,
    pi,
    sqrt,
    ceil,
    randrange,
    choice,
    xgcd,
    reduce,
    ZZ,
    QQ,
    is_square,
    vector,
    matrix,
    identity_matrix,
    random_matrix,
    zero_matrix,
    IntegralLattice,
    sage_eval,
    ideal,
)

from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionLatticeSampler
from sage.matrix.matrix_integer_dense_hnf import hnf_with_transformation

f = 7  # Matrix challenges are randomly sampled with coefficients in [0, f-1].
c = 1 - ((1 + e ** (-pi)) ** (-1))

# ----------------
def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        # Call the function being decorated and return the result
        return func(*args, **kwargs)
    wrapper.count = 0
    # Return the new decorated function
    return wrapper

# -------------------------------
def arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument('-n',  '--dimension', type=int, help='Matrix dimension: n-by-n matrices', required=True)
    parser.add_argument('-gb', '--groebner_basis', action='store_true', help='Groebner basis approach')
    parser.add_argument('-v',  '--verbose', action='store_true', help='verbose help')
    parser.add_argument('-r',  '--number_of_samples', type=int, help='Number of samples to be used in the Groebner basis approach')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    options = parser.parse_args(args)
    return options

# --------------------------------------
def print_matrix(m: matrix, label: str):
    """

    :param m: matrix with integer coefficients
    :param label: name of the matrix variable
    """
    print(f'\n{label} :\n{m}')


# ----------------------------------------------------
# Decorate group_action() with the counter() decorator
@counter
def group_action(u: matrix, q: matrix):
    """
    Inner product concerning the matrix Q
    :param u: unimodular matrix U over ZZ
    :param q: Gram matrix, which gives a quadratic form
    :return: Uᵀ×Q×U
    """
    return u.transpose() * q * u


# ----------------------------------
def sampling_quadratic_form(n: int, verbose=True):
    """
    Sampling a random positive definite matrix Q with integer coefficients in [-f^2n, f^2n]
    :param n: matrix dimension
    :param verbose:
    :return: Positive definite matrix Q
    """

    bad = 0
    print('')
    while True:
        b = random_matrix(ZZ, n, n, x=0, y=f)
        while b.det() == 0:
            b = random_matrix(ZZ, n, n, x=0, y=f)
        q = b.transpose() * b
        if verbose and n <= 16:
            print(f'\r# Random lattices constructed with non-trivial automorphism:\t{bad}', end='')
            sys.stdout.flush()
            bad += 1
            aut = IntegralLattice(q)
            aut = aut.automorphisms()
            if len(aut.list()) == 2:
                break
        else:
            # Computing the automorphism group of a lattice is expensive (exponential concerning n).
            # We assume random lattices has trivial automorphism with high probability (when benchmarking)
            break

    assert (q.is_positive_definite())
    assert q.is_symmetric()
    assert (q.det() != 0)
    if verbose and n <= 16:
        print('\n')
        assert len(aut.list()) == 2
        return q, lambda: matrix(ZZ, aut.random_element())
    else:
        # Computing the automorphism group of a lattice is expensive.
        # We assume random lattices has trivial automorphism with high probability (when benchmarking)
        return q, lambda: choice([identity_matrix(ZZ, n), -identity_matrix(ZZ, n)])


# ---------------------------------------------------------
# Decorate sampling_from_dsq() with the counter() decorator
@counter
def sampling_from_dsq(q: matrix, n: int, s: float):
    """
    Sampling from Dₛ([Q])
    :param q: a quadratic form Q
    :param n: matrix dimension
    :param s: parameter required on the Discrete Gaussian Distribution
    :return: unimodular matrix U, and the quadratic form Uᵀ×Q×U
    """
    m = int(ceil(2 * n / c))
    d = DiscreteGaussianDistributionLatticeSampler(IntegralLattice(q), sigma=s)
    while True:
        y = matrix(ZZ, [d() for _ in range(0, m, 1)]).transpose()
        if y.rank() >= n:
            break

    assert (y.rank() == n)
    t, u = hnf_with_transformation(y)
    assert t == (u * y)
    u = u.inverse()
    det_u = u.det()
    assert det_u == 1 or det_u == -1
    r = u.transpose() * q * u
    return u, r

# ------------------------------------------------------------------
# Decorate sampling_unimodular_matrix() with the counter() decorator
@counter
def sampling_unimodular_matrix(n: int, t=None):
    """

    :param n: matrix dimension
    :param t: integer bound, each entry will be sample from [-t,t]. By default (t is None) we set t=n
    :return: an unimodular matrix with entries uniformly sampled from [-t,t]
    """
    if t is None:
        t = n
    assert(type(t) == int)

    def minors_determinant(v):
        minors_det = []
        for k in range(0, n, 1):
            rows = list(range(0, k, 1)) + list(range(k + 1, n, 1))
            vk = v[rows,:n - 1]
            minors_det.append(vk.det() * (-1)**(n + 1 + k))
        return minors_det

    def euclidean_algorithm(v):
        if 0 in v:
            return 0, []

        (div, u, w) = xgcd(v[0], v[1])
        x = [u, w]
        for k in range(2, n, 1):
            (div, u, w) = xgcd(v[k], div)
            x = [xj * w for xj in x]
            x += [u]

        assert(sum([x[k] * v[k] for k in range(0, n, 1)]) == div)
        assert(reduce(lambda z,z_: z and z_, [vk % div == 0 for vk in v]))
        return div, x

    def least_squares(data: list):
        x = matrix([vector([point[0], 1]) for point in data])
        y = matrix([point[1] for point in data]).transpose()
        return (x.transpose() * x).solve_right(x.transpose()*y)

    m = zero_matrix(ZZ, n, n)
    d = 0
    while d != 1:
        for i in range(0, n, 1):
            for j in range(0, n - 1, 1):
                m[i,j] = randrange(-t, t + 1)
        d, x = euclidean_algorithm(minors_determinant(m))

    assert(d == 1)
    m[:, n - 1] += (matrix(x).transpose() * (-1) ** randrange(0, 2))

    # Least-square step
    c_tilde = []
    for k in range(0, n - 1, 1):
        tmp = m[:, k]
        (dk, ck) = least_squares( [ (tmp[j, 0], x[j]) for j in range(0, n, 1) ] ).transpose().list()
        c_tilde.append(floor(ck / (n - 1.0) + 0.5))

    for k in range(0, n - 1, 1):
        m[:, n - 1] -= (c_tilde[k] * m[:, k])

    assert(m.det()**2 == 1)
    return m.transpose()


# -----------------------------------------------
# Decorate recover() with the counter() decorator
@counter
def recover(q0: list, q1: list, n: int):
    """
    Recovery of the unimodular matrix
    :param q0: list of quadratic forms Q
    :param q1: list of quadratic forms Q' = Vᵀ×Q×V
    :param n: matrix dimension
    :return: the secret unimodular matrix V
    """
    m = n * (n + 1) // 2
    assert len(q0) >= m
    assert len(q1) >= m
    q0_copy = deepcopy(q0)
    q = []
    q_ = []

    for i in range(0, m, 1):
        for j in range(0, n - 1, 1):
            q0_copy[i].rescale_row(j, 2, j + 1)
        for j in range(0, n, 1):
            q += q0_copy[i][j][j:].list()
        q_ += q1[i].diagonal()

    q = matrix(QQ, m, m, q)
    q_ = matrix(QQ, m, n, q_)

    if q.rank() != m:
        # Handle case when we did not reach m linear independent equations (it only occurs for n = 2... to small case)
        return False, None

    # solution = (q.inverse() * q_)[:n, :n]
    solution = q.solve_right(q_)[:n, :n]

    if 0 in solution[0,:].list():
        # Handle if there is a zero in the first column.
        return True, None

    assert not (0 in solution[0,:].list())
    candidate_v = [[0] * n] * n
    z = 0
    for i in range(0, n, 1):
        assert is_square(solution[0, i])
        v_pivot = ZZ(sqrt(solution[0, i]))
        tmp = solution[:, i] / v_pivot
        candidate_v[i] = tmp .list()

    # Choose the right sign of each column via the norm equations
    for i in range(1, n):
        # plus and minus original column solution
        ith_column = matrix(ZZ, n, 1, candidate_v[i])
        if matrix(ZZ, 1, n, candidate_v[i - 1]) * q0[0] * ith_column == q1[0][i - 1,i]:
            candidate_v[i] = ith_column.list()
        else:
            assert matrix(ZZ, 1, n, candidate_v[i-1]) * q0[0] * ith_column == -q1[0][i-1,i]
            candidate_v[i] = (-ith_column).list()

    candidate_v = matrix(ZZ, n, n, candidate_v).transpose()

    return True, candidate_v


# -----------------------------------------------
# Decorate recover_gb() with the counter() decorator
@counter
def recover_gb(q0: list, q1: list, n: int):
    """
    Recovery of the unimodular matrix but using Gröbner Base
    :param q0: list of quadratic forms Q
    :param q1: list of quadratic forms Q' = Vᵀ×Q×V
    :param n: matrix dimension
    :return: the secret unimodular matrix V
    """
    # Lists of column equations
    eqs = []
    for i in range(n):
        eqs.append([])

    def polyring_str(n):
        # Constructs a string that evaluates into a polynomial ring over QQ with n variables
        res = "PolynomialRing(QQ, names=["
        for i in range(n):
            res += "'x{}',".format(i)
        res += "])"
        return res

    polyring = sage_eval(polyring_str(n))
    v = vector(polyring, polyring.variable_names())

    for i in range(len(q0)):
        q0_i = q0[i]
        q1_i = q1[i]
        eq = v * q0_i * v
        for j in range(n):
            eqs[j].append(eq - q1_i[j,j])
    
    # GB extract
    sols = []
    print('')
    for i in range(n):
        I = ideal(eqs[i])
        sys.stdout.write("\r# Call msolve: {}/{}".format(i + 1, n))
        sys.stdout.flush()
        sols.append(I.variety(algorithm="msolve",proof=False))
    print('')

    def extract_vec_from_point(pt):
        res = [0] * n
        for k in pt.keys():
            idx = int(str(k)[1:])
            res[idx] = pt[k]
        return vector(res)

    # Choose the right sign of each column via the norm equations
    candidate_v = [[0] * n] * n
    candidate_v[0] = extract_vec_from_point(sols[0][0])
    for i in range(1, n):
        # plus and minus original column solution
        ith_column = matrix(ZZ, n, 1, extract_vec_from_point(sols[i][0]))
        if matrix(ZZ, 1, n, candidate_v[i - 1]) * q0[0] * ith_column == q1[0][i - 1,i]:
            candidate_v[i] = ith_column.list()
        else:
            assert matrix(ZZ, 1, n, candidate_v[i-1]) * q0[0] * ith_column == -q1[0][i-1,i]
            candidate_v[i] = (-ith_column).list()

    candidate_v = matrix(ZZ, n, n, candidate_v).transpose()
    return True, candidate_v


# --------------------------------------------------------------------
def get_n_quadratic_forms(q0: matrix, p1: matrix, p2: matrix, n: int):
    """
    Get quadratic forms of the form: P1×(Q×P1)^k = Vᵀ×(Q×[P2×Q]^k)×V
    :param q0: a quadratic form Q
    :param p1: the quadratic form Vᵀ×Q×V
    :param p2: the quadratic form V×Q×Vᵀ
    :param n: matrix dimension
    :return: the set of n linearly independent quadratic forms
    """
    q_p1 = q0 * p1
    p2_q = p2 * q0
    output = [p1]
    inside = [q0]
    for k in range(0, n - 1, 1):
        output.append(output[k] * q_p1)
        inside.append(inside[k] * p2_q)

    return output, inside


# ---------------------------------------------------------------------------------------------------
def algorithm(q: matrix, n: int, s: float, function, oracle_calls, optimized=False, Recover=recover, Groebner=False, samples=None):
    """
    Algorithm simulation
    :param q: a quadratic form Q
    :param n: matrix dimension
    :param s: parameter required on the Discrete Gaussian Distribution
    :param function: determines the function concerning TQFP or IQFP computation
    :param oracle_calls: simulation of the oracle call concerning TQFP or IQFP
    :param optimized: optimization flag
    :return: the secret unimodular matrix V
    """

    if not optimized:
        (oracle_call,) = oracle_calls
    else:
        (oracle_call, second_oracle_call) = oracle_calls
        steps = (n + 1)
        steps += (steps % 2)

    m = n * (n + 1) // 2
    v = identity_matrix(ZZ, n)  # First equation comes from the public key Q' = Uᵀ×Q×U (i.e., V = Identity)
    candidate_v = None
    full_rank = False
    loop = 0
    print('')
    while not full_rank:
        loop += 1
        q0 = []
        q1 = []
        if not optimized:
            for i in range(0, m, 1):
                sys.stdout.write("\r# [#{}] Calling to the oracle: {}/{}".format(loop, i + 1, m))
                sys.stdout.flush()
                q0.append(function(v, q))
                q1.append(oracle_call(v))
                v, _ = sampling_from_dsq(q, n, s)
        else:
            if not Groebner:
                # Trick as in Remark 7
                number_of_samples = steps // 2
            else:
                # Trick as in Remark 8 (Gröbner basis optimization)
                if samples is None:
                    number_of_samples = n.bit_length()
                else:
                    number_of_samples = samples
            
            for i in range(0, number_of_samples, 1):
                sys.stdout.write("\r# [#{}] Calling to the oracle: {}/{}".format(loop, i + 1, number_of_samples))
                sys.stdout.flush()
                p0 = function(v, q)
                p1 = oracle_call(v)
                p2 = second_oracle_call(v)
                q1_list, q0_list = get_n_quadratic_forms(p0, p1, p2, n)
                for j in range(0, n, 1):
                    q0.append(q0_list[j])
                    q1.append(q1_list[j])
                v, _ = sampling_from_dsq(q, n, s)

        full_rank, candidate_v = Recover(q0, q1, n)

    assert full_rank

    # To recover UR
    r = identity_matrix(ZZ, n)
    while candidate_v is None:
        r = sampling_unimodular_matrix(n)
        assert(r.det() ** 2 == 1)
        q1_ = [ group_action(r, q1_k) for q1_k in q1 ]
        full_rank, candidate_v = Recover(q0, q1_, n)
        assert full_rank

    candidate_v = candidate_v * r.inverse()

    assert candidate_v.det() ** 2 == 1
    return candidate_v

def create_to_csv_file(file_name: str, data_list: list):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        # field = ["n", "Sampling", "Recover", "Recover GB #1", "Recover GB #2", "Recover GB #3"]
        field = ["n", "Sampling", "Recover", "Recover GB"]
        writer.writerow(field)
        for data_row in data_list:
            writer.writerow(data_row.values())

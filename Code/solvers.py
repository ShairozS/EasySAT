'''
A set of classes and functions to solve k-sat instances
'''

import itertools



def __select_literal(cnf):
    for c in cnf:
        for literal in c:
            return literal[0]


def brute_force(cnf):
    literals = set()
    for conj in cnf:
        for disj in conj:
            literals.add(disj[0])
 
    literals = list(literals)
    n = len(literals)
    steps = 0
    for seq in itertools.product([True,False], repeat=n):
        steps += 1
        a = set(zip(literals, seq))
        if all([bool(disj.intersection(a)) for disj in cnf]):
            return True, a, steps
 
    return False, None, steps


def dpll(cnf, assignments={}, steps = 1):
    
    if len(cnf) == 0:
        return True, assignments, steps
 
    if any([len(c)==0 for c in cnf]):
        return False, None, steps
 
    l = __select_literal(cnf)
    
    new_cnf = [c for c in cnf if (l, True) not in c]
    new_cnf = [c.difference({(l, False)}) for c in new_cnf]
    sat, vals, steps = dpll(new_cnf, {**assignments, **{l: True}}, steps + 1)
    if sat:
        return sat, vals, steps
 
    new_cnf = [c for c in cnf if (l, False) not in c]
    new_cnf = [c.difference({(l, True)}) for c in new_cnf]
    sat, vals, steps = dpll(new_cnf, {**assignments, **{l: False}}, steps + 1)
    if sat:
        return sat, vals, steps
 
    return False, None, steps
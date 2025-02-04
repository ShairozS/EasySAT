'''
A set of classes and functions to solve k-sat instances
'''

import itertools
import numpy as np
from pysat import solvers

def count_solutions(instance, solver = None, return_solutions = False):
    '''
    Count the number of solutions to a k-sat instance using
    a SAT solver. This function utilizes a simple brute
    force approach using a solve-exclude loop and is not 
    recommended for larger instances. 
    '''
    if solver is None:
        solver = solvers.Glucose3()
    for clause in instance:
        solver.add_clause(clause)

    solutions = []
    while solver.solve():
        solution = solver.get_model()
        solutions.append(solution)
        # Add a clause to prevent the solver from finding the same solution again
        solver.add_clause([-lit for lit in solution])

    if return_solutions:
        return len(solutions), solutions
    return len(solutions)
    
def find_backbone(instance, solver=None):
    '''
    Return the set of backbone literals for a
    SAT instance if it is solvable.
    '''
    if solver is None:
        solver = solvers.Glucose3()
        
    # First, get all the solutions to the instance
    sat, solutions = count_solutions(instance, solver=solver, return_solutions = True)
    if sat == 0:
        raise NotSatisfiable
    backbone = []
    
    # Now, iterate through the solutions to find variables that only show up with a single polarity
    unique_polarities = np.apply_along_axis(lambda x: len(np.unique(x)), 0, s)
    
    backbone_literals = np.where(unique_polarities < 2)[0]
    if len(backbone_literals) > 0:
        for b in backbone_literals:
            backbone.append(solutions[0][b])

    return(backbone)


def str_to_kcnf(formula):
    '''
    Convert a KSAT formula in string format to CNF format (i.e [[1,-2], [2, -3]...])
    '''
    sign = lambda x: int(x[0]) if x[1] else -int(x[0])
    conjuncts = []
    for c in formula:
        conj = list(c)
        conj_formatted = [sign(x) for x in conj]
        conjuncts.append(conj_formatted)

    return(conjuncts)

def kcnf_to_str(formula):
    '''
    Convert a KSAT formula in CNF format to string format (i.e (1, True), (2, False) ...)
    '''
    sign = lambda x: (str(abs(x)), True) if x>0 else (str(abs(x)), False)
    conjuncts = []
    for c in formula:
        conj = list(c)
        conj_formatted = {sign(x) for x in conj}
        conjuncts.append(conj_formatted)

    return(conjuncts)
    

def test_conj(conj, model):
    '''
    Test an individual conjunct against a model

    Example:
    conj = [-2, 1, 3]
    model = [1,0,1]
    test_conj(conj, model)
    > True
    '''
    agreement = []
    for c in conj:
        pred = model[abs(c)-1]
        if pred<=0 and c<0:
            agreement.append(1)
        elif pred==1 and c>0:
            agreement.append(1)
    if len(agreement) == 0:
        return False
    return(max(agreement) == 1)


def attempt(formula, model):
    ## Attempt to solve a SAT formula with a proposed model
    solved = []
    for conj in formula:
        solved.append(test_conj(conj, model))
    return(min(solved)==True)


class DPLL:


    def __init__(self):
        self.formula = None
        self.stats = {'propagations':0}

    
    def append_formula(self,cnf):
        self.formula = kcnf_to_str(cnf)

    @staticmethod
    def __select_literal(cnf):
        for c in cnf:
            for literal in c:
                return literal[0]
            
    def dpll(self, cnf, assignments={}, steps = 1):
    
        if len(cnf) == 0:
            return True, assignments, steps
     
        if any([len(c)==0 for c in cnf]):
            return False, None, steps
     
        l = self.__select_literal(cnf)
        
        new_cnf = [c for c in cnf if (l, True) not in c]
        new_cnf = [c.difference({(l, False)}) for c in new_cnf]
        sat, vals, steps = self.dpll(new_cnf, {**assignments, **{l: True}}, steps + 1)
        if sat:
            return sat, vals, steps
     
        new_cnf = [c for c in cnf if (l, False) not in c]
        new_cnf = [c.difference({(l, True)}) for c in new_cnf]
        sat, vals, steps = self.dpll(new_cnf, {**assignments, **{l: False}}, steps + 1)
        if sat:
            return sat, vals, steps
     
        return False, None, steps

    def solve(self,):
        sat, model, props = self.dpll(self.formula)
        self.stats['propagations'] = props
        # Annoying manipulation to format output model correctly
        if sat:
            truth_vals = [int(model[x]) if model[x]==1 else -1 for x in model]
            literals = [int(x) for x in model]
            model = sorted([truth_vals[i]*literals[i] for i in range(len(literals))], key = abs)
        return(sat, model, self.stats['propagations'])


class BruteForce:


    def __init__(self):
        self.formula = None
        self.stats = {'propagations':0}
        
    def append_formula(self,cnf):
        self.formula = cnf
        self.num_of_literals = max([max([abs(z) for z in x]) for x in self.formula])
        
    def solve(self):

        solved = False
        gen = np.random.Generator(np.random.PCG64())
        tried = []
        while not solved:

            sample = gen.choice(a = [-1,1], size = self.num_of_literals)
            sample = list(sample)
            if sample not in tried:
                
                # add to propagations
                self.stats['propagations'] += 1
                
                # Does this satisfy? 
                solved = attempt(self.formula, sample)
                
                # If not, add to tried combinations
                tried.append(sample)
                
            # If # of propagations == number of potential models, exit w/ UNSAT
            if self.stats['propagations'] >= (2**(self.num_of_literals)):
                return False, None, self.stats['propagations']

        sample = [sample[x]*(x+1) for x in range(self.num_of_literals)]
        return True, sample, self.stats['propagations']




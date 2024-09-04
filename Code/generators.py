'''
A set of classes and functions to generate random k-sat instances
'''

import string
import itertools
import random
import numpy as np

import random 
import string
import itertools
from collections import Counter
class KSAT_Generator:


    def __init__(self, max_literals = 100):
        self.var_map = {}
        letters = list(string.ascii_uppercase)
        letters2 = itertools.combinations(letters, 2)
        while(len(letters)) < max_literals:
            letters.append(''.join(next(letters2)))
        for i in range(1,max_literals):
            self.var_map[i] = letters[i-1]
        #print(self.var_map)

    def random_kcnf(self, n_literals, n_conjuncts, k=3):
        '''
        Generate a random KSAT formula in string form
        '''
        result = []
        for _ in range(n_conjuncts):
            conj = set()
            for _ in range(k):
                index = random.randint(1, n_literals)
                conj.add((
                    str(index),#.rjust(10, '0'),
                    bool(random.randint(0,2)),
                ))
            result.append(conj)
        return result


    def kcnf_to_cnf(self, formula):
        '''
        Convert a random KSAT formula in string format to CNF format
        '''
        new_formula = []
        for clause in formula:
            new_clause = []
            vars = [int(x[0]) for x in clause]
            signs = [1 if x[1] else -1 for x in clause]
            new_clause.append([a*b for a,b in zip(vars, signs)])
            new_formula.append(new_clause[0])
            
        return(new_formula)


    def from_dimacs_file(self, file):
        '''
        Read a DIMACS formatted .cnf file and output in cnf format

        Examples: https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/QG/qg.descr.html
        '''
        with open(samp, 'r') as f:
            lines = f.readlines()
        comments = [l.replace("c", "") for l in lines if l[0]=='c']
        clauses = [l.replace('0\n', "").rstrip().split(" ") for l in lines if (not l[0]=='c' and not l[0]=='p')]
        clauses = [[int(x) for x in x] for x in clauses[:-1]]
        clauses.append(clauses[-1][:-1])
        print(*comments, sep = '\n')
        return(clauses)

    @staticmethod
    def remap_vals(samp_clauses):
        '''
        Remap the values of an array so they're ordinal
        '''
        unique_variables = set(list(itertools.chain(*samp_clauses)))
        
        # Form map
        print("Unique variables: ", unique_variables)
        mapping = {}; idx = 1
        for i in sorted(unique_variables, reverse = True):
            if i==0:
                i += 10000
            # If a positive-negative pair
            if i in unique_variables and -i in unique_variables:
                if i > 0:
                    mapping[i] = idx
                    mapping[-i] = idx
                else:
                    mapping[i] = -idx
                    mapping[-i] = idx
                
            # If only positive
            elif i > 0 and -i not in unique_variables:
                mapping[i] = idx

            # If only negative
            elif i < 0 and -i not in unique_variables:
                mapping[i] = -idx

            idx += 1


        print(mapping)
        # Remap
        remapped_clauses = []
        for clause in samp_clauses:
            new_clause = []
            for i in clause:
                new_clause.append(mapping[i])
            remapped_clauses.append(new_clause)

        return(remapped_clauses)

    #@staticmethod
    def cnf_to_matrix(self,formula):
        '''
        Propositions are rows
        Literals are columns
        Values are true (1) or false (-1) occurance
        '''
        occurrence_count = Counter(chain(*map(lambda x: x, formula)))

        items = list(occurrence_count.keys())  # items, with no repetitions
        all_items = list(occurrence_count.elements())  # all items
        
        
        img = np.zeros((len(formula), len([i for i in items if i > 0]) + 1))
        rmp = self.remap_vals(formula)
        
        for i in range(len(rmp)):
            clause = rmp[i]        
            for var in clause:
                if var < 0:
                    negate = -1
                else:
                    negate = 1
                img[i, abs(var)] = negate
        
        img = img[:, 1:]
        return(img.T)

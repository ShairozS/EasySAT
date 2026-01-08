'''
A set of classes and functions to generate random k-sat instances
'''

import string
import itertools
import random
import numpy as np
from itertools import chain
import random 
import string
import itertools
from collections import Counter



class KSAT_Generator:

    def __init__(self, max_literals=100):
        self.var_map = {}
        letters = list(string.ascii_uppercase)
        letters2 = itertools.combinations(letters, 2)
        while len(letters) < max_literals:
            letters.append(''.join(next(letters2)))
        for i in range(1, max_literals + 1):
            self.var_map[i] = letters[i - 1]


    def random_kcnf(self,
                    n_literals,
                    n_conjuncts=None,
                    k=3,
                    dimacs=True,
                    exactly_k=True,
                    no_contradictions=True):

        # Phase-transition default if not specified
        if n_conjuncts is None:
            n_conjuncts = int(self._critical_ratio(k) * n_literals)

        clauses = self._balanced_kcnf(
            n_vars=n_literals,
            n_clauses=n_conjuncts,
            k=k,
            no_contradictions=no_contradictions
        )

        # Inject hardness (small probability)
        if random.random() < 0.5:
            clauses = self._near_sat_noise(clauses, n_literals, noise=0.05)

        if random.random() < 0.3:
            clauses += self._unsat_core(k)

        return self.kcnf_to_cnf(clauses) if dimacs else clauses


    def _balanced_kcnf(self, n_vars, n_clauses, k, no_contradictions):
        clauses = []
        usage = {i: 0 for i in range(1, n_vars + 1)}

        for _ in range(n_clauses):
            clause = set()
            vars_sorted = sorted(usage, key=lambda v: usage[v])

            for var in vars_sorted:
                if len(clause) == k:
                    break
                sign = random.choice([True, False])
                lit = (var, sign)
                if no_contradictions and (var, not sign) in clause:
                    continue
                clause.add(lit)
                usage[var] += 1

            clauses.append(clause)

        return clauses

    def _near_sat_noise(self, clauses, n_vars, noise=0.05):
        assignment = {i: random.choice([True, False]) for i in range(1, n_vars + 1)}
        new_clauses = []

        for clause in clauses:
            new_clause = set()
            for var, sign in clause:
                if random.random() < noise:
                    sign = not sign
                new_clause.add((var, sign))
            new_clauses.append(new_clause)

        return new_clauses

    def _unsat_core(self, k):
        # Small hard UNSAT core
        if k < 3:
            return []
        return [
            {(1, True), (2, True), (3, True)},
            {(1, False), (2, True), (3, True)},
            {(1, True), (2, False), (3, True)},
            {(1, True), (2, True), (3, False)},
            {(1, False), (2, False), (3, False)}
        ]


    def _critical_ratio(self, k):
        return {
            3: 4.26,
            4: 9.93,
            5: 21.1
        }.get(k, 4.26)

    def kcnf_to_cnf(self, clauses):
        """
        DIMACS CNF formatter
        """
        max_var = max(var for clause in clauses for var, _ in clause)
        lines = [f"p cnf {max_var} {len(clauses)}"]
        for clause in clauses:
            line = []
            for var, sign in clause:
                lit = var if sign else -var
                line.append(str(lit))
            line.append("0")
            lines.append(" ".join(line))
        return "\n".join(lines)


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

    @staticmethod
    def from_dimacs_file(file, print_comments = True):
        '''
        Read a DIMACS formatted .cnf file and output in cnf format

        Examples: https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/QG/qg.descr.html
        '''
        with open(file, 'r') as f:
            lines = f.readlines()
        comments = [l.replace("c", "") for l in lines if l[0]=='c']
        clauses = [l.replace('0\n', "").rstrip().split(" ") for l in lines if (not l[0]=='c' and not l[0]=='p')]
        clauses = [[x for x in clause if x != ''] for clause in clauses]
        clauses = [c for c in clauses if len(c) >= 2]
        clauses = [[int(x) for x in x if int(x) != 0] for x in clauses]
        if print_comments:
            print(*comments, sep = '\n')
        return(clauses)
        

    @staticmethod
    def remap_vals(samp_clauses):
        '''
        Remap the values of an array so they're ordinal
        '''
        unique_variables = set(list(itertools.chain(*samp_clauses)))
        
        # Form map
        #print("Unique variables: ", unique_variables)
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


        #print(mapping)
        # Remap
        remapped_clauses = []
        for clause in samp_clauses:
            new_clause = []
            for i in clause:
                new_clause.append(mapping[i])
            remapped_clauses.append(new_clause)

        return(remapped_clauses)

    def cnf_to_matrix(self,formula):
        '''
        Propositions are rows
        Literals are columns
        Values are true (1) or false (-1) occurance
        '''

        occurrence_count = Counter(chain(*map(lambda x: x, formula)))
        items = list(occurrence_count.keys())  # items, with no repetitions
    
        img = np.zeros((len(formula), 10000))#len([i for i in items if i > 0]) + 1))
        rmp = self.remap_vals(formula)
    
        for i in range(len(formula)):
            clause = formula[i]        
            for var in clause:
                
                if var < 0:
                    negate = -1
                else:
                    negate = 1
                img[i, abs(var)] = negate
        
        img = img[:,~np.all(img == 0, axis = 0)]
    
        self.cnf_mat = img
        return(img)


    def describe_literal(self, formula, literal):
        '''
        Return the positive activity, negative activity, correlated set, and correlations
        of a literal
        '''
        cnf_mat = self.cnf_to_matrix(formula)

        # What props does the literal appear in
        relevent_props = np.where(cnf_mat[:, literal] != 0)[0]
        cor_set = []
        ent_set = []
        pos_activity = 0; neg_activity = 0
        
        for prop in relevent_props:
            p = cnf_mat[prop,:]
            
            if p[literal] == 1:
                pos_activity += 1
                cor_set += list(np.where(p==1)[0])
                
            elif p[literal] == -1:
                neg_activity += 1
                ent_set += list(np.where(p==1)[0])
                
        ent_set = set(ent_set)
        cor_set = set(cor_set)

        out = {'pos_activity':pos_activity,
               'neg_activity':neg_activity,
               'correlations':cor_set,
               'entanglements': ent_set}
        return(out)
        
        
    def cnf_score(self, formula):

        if self.cnf_mat is None:
            cnf_mat = self.cnf_to_matrix(formula)
        else:
            cnf_mat = self.cnf_mat
            
        # For each row
        statements, literals = cnf_mat.shape
        for i in range(statements):
            score = 0

            # What variables are positive
            positives = np.where(cnf_mat[i, :] == 1)[0]
            
            # How many other statements involve the positive variables
            for p in positives:
                #score += np.sum(np.abs(cnf_mat[:, p])) - 1 / statements
                score += np.sum(np.clip(cnf_mat[:, p], 0, 1)) - 1 / statements
        return(score)
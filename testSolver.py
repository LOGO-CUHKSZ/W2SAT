from pysat.solvers import Glucose3
from pysat.formula import CNF

import os
import time


filenames = os.listdir('./dataset/train_formulas/')

skip_table = ['aes_24_4_keyfind_2.processed.cnf',
              'sat_prob_83.processed.cnf', 'aes_64_1_keyfind_1.processed.cnf',
              '']
# skip_table = []

test_filenames = [x for x in filenames if 'aes' not in x and 'sat' not in x]
for filename in test_filenames:
    start_time = time.time()
    print(filename)
    formula = CNF(from_file=f"./dataset/train_formulas/{filename}")
    clauses = formula.clauses
    solver = Glucose3(bootstrap_with=formula.clauses)
    result = solver.solve()
    print(
        f'- {filename}: {result}, {len(clauses)}, {formula.nv}, {time.time() - start_time}')

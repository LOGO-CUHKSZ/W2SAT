import numpy as np
import random as rd


def read_sat(sat_path):
    with open(sat_path) as f:
        sat_lines = f.readlines()
        header = sat_lines[0]
        header_info = header.replace("\n", "").split(" ")
        vars_num = int(header_info[-2])
        clauses_num = int(header_info[-1])

        sat = [
            [int(x) for x in line.replace(" 0\n", "").split(" ")]
            for line in sat_lines[1:]
        ]

        return vars_num, clauses_num, sat

vars_num, clauses_num, sat = read_sat("~/Workspace/W2SAT/result/generation/aes_32_3_keyfind_2.processed.cnf/sample-20.cnf")

def unsatcore_detect (vars_num, sat):
    sat_set = dict()
    unsat_cores = []
    for i, clause in enumerate(sat):
        sat_set[tuple(clause)] = i
    for index, clause in enumerate(sat):
        len_clause = len(clause)
        state_number = 1 << len_clause
        new_clause = clause.copy()
        flag = 1
        unsat_core = []
        for state in range(state_number): # e.g. for clause(1,2,3), enumerate the sign of it, in total 8 cases.
            for i in range(len_clause):
                new_clause[i] = clause[i] if state >> i & 1 else -clause[i]
            if tuple(new_clause) not in sat_set:
                flag = 0
                break
            unsat_core.append(sat_set[tuple(new_clause)])
        if flag:
            unsat_cores.append(sorted(unsat_core))
            for i in unsat_core:
                del sat_set[tuple(sat[i])]
    return unsat_cores

def unsatcore_fix (vars_num, sat, unsatcore):
    rd.seed(114)
    for core in unsatcore:
        print(core)
        clause_index = core[rd.randint(0, len(core) - 1)]
        new_literal = sat[clause_index][0]
        while new_literal in sat[clause_index]:
            new_literal = rd.randint(0, vars_num - 1)
        new_literal = -new_literal if rd.random() < 0.5 else new_literal
        #print(new_literal,  sat[clause_index])
        sat[clause_index].append(new_literal)
        sat[clause_index] = sorted(sat[clause_index])
        #print(sat[clause_index])

core = unsatcore_detect(vars_num, sat)
unsatcore_fix(vars_num, sat, core)
#print(unsatcore_detect(vars_num, sat)) # expected to be []

filename = 'b_out.cnf'
with open(filename, "w") as f:
    f.write(f"p cnf {vars_num} {clauses_num}\n")
    for clause in sat:
        f.write(f"{' '.join([str(v) for v in clause])} 0\n")

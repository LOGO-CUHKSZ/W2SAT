import os
from multiprocessing import Pool
import itertools
import subprocess
import time


def gridSearch(v, c, instance):
    start_time = time.time()
    for i in range(2):
        path = f"../dataset/formulas/{instance}"
        subprocess.call(
            [
                "../eval/glucose",
                f"-var-decay={v}",
                f"-cla-decay={c}",
                path,
            ],
            stdout=subprocess.DEVNULL,
        )
    solve_time = time.time() - start_time
    solve_time = solve_time / 2
    log = open(f"../result/glucose/{instance}-{v:.3f}-{c:.3f}.log", "w")
    log.write(f"{v:.3f},{c:.3f},{instance},{solve_time:.3f}")


if __name__ == "__main__":
    vdecays = [0.75, 0.80, 0.85, 0.90, 0.95]
    cdecays = [0.7, 0.8, 0.9, 0.99, 0.999]
    instances = [
        "sat_prob_83.processed.cnf",
        "aes_32_3_keyfind_2.processed.cnf",
        "countbitsrotate016.processed.cnf",
        "cmu-bmc-longmult15.processed.cnf",
        "smulo016.processed.cnf",
        "countbitssrl016.processed.cnf",
    ]
    # instances = ["smulo016.processed.cnf", "countbitssrl016.processed.cnf"]
    p = Pool(30)
    for parameters in itertools.product(vdecays, cdecays, instances):
        print(parameters)
        p.apply_async(
            gridSearch,
            args=(
                parameters[0],
                parameters[1],
                parameters[2],
            ),
        )
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")

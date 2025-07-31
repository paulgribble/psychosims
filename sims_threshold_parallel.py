import os
import tempfile
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from multiprocessing import Process, Queue, cpu_count
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' # Set this too just in case

S = np.arange(5, 21, 3)
N = np.arange(5, 16, 2)
E = np.arange(0.1, 3.1, 0.1)
X = 10000

param_grid = list(product(range(len(S)), range(len(N)), range(len(E))))

# Split param_grid into chunks for each process
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def worker(params_chunk, queue, worker_id):
    for idx, (i_s, i_n, i_e) in enumerate(params_chunk):
        b1 = 0.4
        b0 = -b1 * E[i_e]
        num_subs = S[i_s]
        num_trials = N[i_n]

        with tempfile.NamedTemporaryFile(delete=False) as fpre, tempfile.NamedTemporaryFile(delete=False) as fpost:
            prefile = fpre.name
            postfile = fpost.name

        try:
            os.system(f"./sims {num_subs * X} {num_trials} 0.0 {b1} > {prefile}")
            os.system(f"./sims {num_subs * X} {num_trials} {b0} {b1} > {postfile}")
            pre = np.genfromtxt(prefile)
            post = np.genfromtxt(postfile)

            count05 = 0
            count01 = 0

            for i_x in range(X):
                i1, i2 = i_x * num_subs, (i_x + 1) * num_subs
                t, pval = ttest_rel(post[i1:i2, 2], pre[i1:i2, 2])
                count05 += int((pval < 0.05) and (t > 0))
                count01 += int((pval < 0.01) and (t > 0))

            result = (i_s, i_n, i_e, count05 / X, count01 / X)
            queue.put(('progress', worker_id, idx + 1, len(params_chunk)))
            queue.put(('result', result))
        finally:
            os.remove(prefile)
            os.remove(postfile)

def main():
    D05 = np.zeros((len(S), len(N), len(E)))
    D01 = np.zeros((len(S), len(N), len(E)))

    num_workers = min(16, cpu_count())
    chunks = chunkify(param_grid, num_workers)
    queue = Queue()

    processes = []
    for i in range(num_workers):
        p = Process(target=worker, args=(chunks[i], queue, i))
        p.start()
        processes.append(p)

    pbars = [tqdm(total=len(chunks[i]), position=i, desc=f"Worker {i}", leave=True) for i in range(num_workers)]

    completed = 0
    while completed < len(param_grid):
        msg = queue.get()
        if msg[0] == 'progress':
            _, wid, done, total = msg
            pbars[wid].n = done
            pbars[wid].refresh()
            completed += 1
        elif msg[0] == 'result':
            _, (i_s, i_n, i_e, d05, d01) = msg
            D05[i_s, i_n, i_e] = d05
            D01[i_s, i_n, i_e] = d01

    for p in processes:
        p.join()

    print("done: simulated %d subjects" % (len(S) * len(N) * len(E) * X))

    plt.figure(figsize=(16, 8))
    for i in range(len(S)):
        plt.subplot(2, 3, i + 1)
        for j in range(len(N)):
            plt.plot(E, D05[i, j, :], '-')
        plt.ylim(0, 1)
        plt.xlim(min(E), max(E))
        plt.title(f"{S[i]} Subjects")
        plt.grid(True)
        if i > 2:
            plt.xlabel("Threshold shift (mm)")
        if i in (0, 3):
            plt.ylabel("Power at p<0.05")
        if i == 5:
            plt.legend(N, loc=4, title="Trials")
    plt.savefig("psychosims.pdf")

if __name__ == "__main__":
    main()

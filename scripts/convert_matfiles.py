import sys
import os
import numpy as np
from scipy.io import loadmat


def convert(file_name, in_folder, coco_folder, out_folder):
    print('converting:', file_name)
    path = in_folder + '/' + file_name
    coco_path = coco_folder + '/' + file_name
    mat = loadmat(path)
    coco_mat = loadmat(coco_path)

    instances_cnt = mat['cmaes_out'].shape[1]

    for i in range(instances_cnt):
        save_path = out_folder + '/' + file_name[:-4] + '_{}.npz'.format(i)
        if os.path.isfile(save_path):
            print('Exists: ', save_path)
            return

        instance = mat['cmaes_out'][0][i][0][0][0][0]

        generation_starts = instance[0][0]
        arxvalids = instance[1].T
        fvalues = instance[2][0]
        orig_evaled = instance[9][0].astype(bool)
        iruns = instance[15][0]
        evals = instance[17][0][0]
        gen_split = generation_starts - 1

        run = coco_mat['runs'][0][i][0]
        coco = np.fromiter((row[0][0] for row in run), np.float64)

        p, d = arxvalids.shape
        g = generation_starts.shape[0]
        print('\tD={}\tP={}\tG={}'.format(d, p, g))

        np.savez(
            save_path,
            points=arxvalids,
            fvalues=fvalues,
            orig_evaled=orig_evaled,
            gen_split=gen_split,
            iruns=iruns,
            evals=evals,
            coco=coco,
        )


if __name__ == "__main__":
    try:
        _, in_folder, coco_folder, out_folder = sys.argv
    except:
        print("Usage python convert_matfiles.py coco_path in_path out_path")
        exit(1)

    mat_files = [f for f in os.listdir(in_folder) if f.endswith('.mat') and f.startswith('exp_doubleEC_28_log_nonadapt')]
    for file in mat_files:
        convert(file, in_folder, coco_folder, out_folder)

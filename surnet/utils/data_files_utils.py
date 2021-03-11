import os
import re



def decode_filename(filename):
    keys = ['dim', 'func', 'kernel']

    # 5 x 24 x 9
    params = [
        [2, 3, 5, 10, 20],
        list(range(1, 24 + 1)),
        ['LIN', 'QUAD', 'SE', 'MATERN5', 'RQ', 'NN(ARCSIN)', 'ADD', 'SE+QUAD', 'GIBBS'],
    ]

    def decode_task_id(task_id):
        ti = task_id - 1

        sizes = [len(p) for p in params]
        k = (len(params)) * [0]

        for i in range(len(params) - 1, -1, -1):
            k[i] = ti % sizes[i]
            ti //= sizes[i]

        return k

    task_id = int(filename.split('_')[-2])
    indicies = decode_task_id(task_id)
    settings = [p[i] for p, i in zip(params, indicies)]

    return dict(zip(keys, settings))


def _file_filter(**filters):
    def f(filename):
        d = decode_filename(filename)
        for k, v in filters.items():
            if d[k] != v:
                return False
        return True

    return f


def get_data_files(folder_path, **filters):
    files = os.listdir(folder_path)
    # file_dicts = [_decode_filename(fn) for fn in files]
    return filter(_file_filter(**filters), files)


def load_generation(experiment, gen):
    points = experiment['points']
    fvalues = experiment['fvalues']
    coco = experiment['coco']
    gen_split = experiment['gen_split']
    orig_evaled = experiment['orig_evaled']

    pos, next_pos = gen_split[gen:gen + 2]
    orig = orig_evaled[:pos]
    x_fit = points[:pos][orig]
    y_fit_base = fvalues[:pos][orig]
    y_fit_coco = coco[:pos][orig]

    orig = orig_evaled[pos:next_pos]
    x_eval = points[pos:next_pos][~orig]
    y_eval_base = fvalues[pos:next_pos][~orig]
    y_eval_coco = coco[pos:next_pos][~orig]

    return x_fit, y_fit_base, y_fit_coco, x_eval, y_eval_base, y_eval_coco

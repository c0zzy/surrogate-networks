import os
import json
import subprocess


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


params_template = """
{
  "model": {
    "name": "deep_gp",
    "noise": 1e-4,
    "arch": "low",
    "kernel": {
      "name": "rbf",
      "scale": true,
      "params": {
        "lengthscale": 1.0,
        "lengthscale_bounds": [1e-2, 100]
      }
    }
  },
  "optimizer": {
    "name": "adam",
    "iters": 1000,
    "lr": 0.005
  },
  "run": {
    "dimensions": [2],
    "functions": null,
    "kernels": null,
    "instances": null
  }
}
"""

params_template_gp = """
{
  "model": {
    "name": "gp",
    "noise": 1e-4,
    "arch": "low",
    "kernel": {
      "name": "rbf",
      "scale": true,
      "params": {
        "lengthscale": 1.0,
        "lengthscale_bounds": [1e-2, 100]
      }
    }
  },
  "optimizer": {
    "name": "adam",
    "iters": 1000,
    "lr": 0.005
  },
  "run": {
    "dimensions": [2],
    "functions": null,
    "kernels": null,
    "instances": null
  }
}
"""

job_template = """
. /storage/brno2/home/kozajan/venv/bin/activate

cd {}

python ../../../sn-code/surnet/run_exps.py -cfg-env=metacentrum -n-cpus=$PBS_NCPUS -params-path={} -job={} > out.log
"""

jobs_dir = '/storage/brno2/home/kozajan/sn-results/jobs/'
folders = os.listdir(jobs_dir)

next_job_num = 0
if folders:
    next_job_num = max([int(f) for f in folders]) + 1

# kernels = ['lin', 'quad', 'rbf', 'matern', 'rq', 'se_quad']
# nums = [1, 2, 3, 4, 5, 8]

k = 'spectral_mixture'
nums = [1, 2, 3, 4, 5, 6, 7, 8, 0]

for dim in [2, 3, 5, 10, 20]:
    # for k, n in zip(kernels, nums):
    for n in nums:
        # for arch in [None]:
        for arch in ['low', 'high']:
            params = json.loads(params_template)
            params['model']['kernel']['name'] = k
            if n:
                params['run']['kernels'] = [n]
            params['model']['arch'] = arch

            params['run']['dimensions'] = [dim]

            job_id = str(next_job_num).zfill(4)
            job_path = jobs_dir + job_id
            os.mkdir(job_path)

            job_params_path = job_path + "/params.json"
            with open(job_params_path, "w") as params_file:
                json.dump(params, params_file, indent=4)

            job_script_name = "sn{}.sh".format(job_id)
            job_script_path = job_path + '/' + job_script_name
            with open(job_script_path, "w") as job_file:
                job_file.write(job_template.format(job_path, job_params_path, job_id))

            with cd(job_path):
                hours = dim
                qsub = 'qsub -l select=1:ncpus=16:mem=8gb -l walltime={}:00:00 {}'.format(hours, job_script_name)
                print(qsub)
                subprocess.Popen(qsub, shell=True)

            next_job_num += 1

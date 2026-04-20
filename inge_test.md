
Temp file 

```
tmux 
bsub -Is -M 32G -n 1 -W 05:00 -gpu "num=1/task:mode=exclusive_process" /bin/bash
conda create -n jupyterlab-debugger2 -c conda-forge jupyterlab=3 "ipykernel>=6" python=3.10
conda activate jupyterlab-debugger2 
pip install vllm[all]
jupyter lab --no-browser --port=5678 --ip=$(hostname -f)   
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python ./demo_act_steer.py
```


```
python3.12 -m build  
python3.12 -m twine upload --repository testpypi dist/*  --verbose
```
python -m pip install --upgrade pip
python -m pip install --upgrade tqdm matplotlib "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax optax orbax-checkpoint torch --index-url https://download.pytorch.org/whl/cpu

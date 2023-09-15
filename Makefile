download:
	wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz -O data/tiny_nerf_data.npz

test:
	pytest .

format:
	find . -name '*.py' -print0 | xargs -0 yapf -i
	isort --atomic **/*.py

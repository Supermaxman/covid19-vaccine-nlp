
import os
from typing import List

from setuptools import setup, find_packages

_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> List[str]:
	with open(os.path.join(path_dir, file_name)) as file:
		lines = [ln.strip() for ln in file.readlines()]
	requirements = []
	for ln in lines:
		# filer all comments
		if comment_char in ln:
			ln = ln[: ln.index(comment_char)].strip()
		# skip directly installed dependencies
		if ln.startswith("http") or "@http" in ln:
			continue
		if ln:
			requirements.append(ln)
	return requirements


VERSION = '0.4.0'
DESCRIPTION = 'NLP package for pytorch and pytorch_lightning with pre-built models'
LONG_DESCRIPTION = 'NLP package for pytorch and pytorch_lightning with pre-built models'

# Setting up
setup(
	name="pytorch-gleam",
	version=VERSION,
	author="Maxwell Weinzierl",
	author_email="maxwellweinzierl@gmail.com",
	description=DESCRIPTION,
	long_description=LONG_DESCRIPTION,
	install_requires=_load_requirements(_PATH_ROOT),
	keywords=['pytorch', 'torch', 'pytorch_lightning', 'nlp', 'deep learning'],
	python_requires=">=3.6",
	license="Apache-2.0",
	packages=find_packages(),
	classifiers=[
		"Environment :: Console",
		"Natural Language :: English",
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Developers",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Information Analysis",
		"License :: OSI Approved :: Apache Software License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.9",
	],
	entry_points={
		"console_scripts": [
			"free-gpus=pytorch_gleam.gpu.free_gpus:main",
			"request-gpus=pytorch_gleam.gpu.request_gpus:main",
			"ex-queue=pytorch_gleam.exqueue.exqueue:main",
			"ex-bqueue=pytorch_gleam.exqueue.exbqueue:main",
			"ex-start=pytorch_gleam.exqueue.exstart:main",
			"ex-stat=pytorch_gleam.exqueue.exstat:main",
			"ex-rm=pytorch_gleam.exqueue.exrm:main",
			"t-parse=pytorch_gleam.parse.tparse:main",
			"f-parse=pytorch_gleam.parse.fparse:main",
			"e-parse=pytorch_gleam.parse.eparse:main",
			"gleam-search-tweet-index=pytorch_gleam.search.tweet_index_search:main",
			"gleam-rerank=pytorch_gleam.search.rerank:main",
			"gleam-rerank-format=pytorch_gleam.search.rerank_format:main",
			"gleam-tweet-to-jsonl=pytorch_gleam.search.tweet_to_jsonl:main",
			"gleam-parse-raw-tweet=pytorch_gleam.parse.parse_raw_tweet:main",
			"gleam-search-candidates=pytorch_gleam.search.select_candidates:main",
		],
	},
)

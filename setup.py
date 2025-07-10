import os
import re

try:
    from setuptools import Extension, find_packages, setup  # noqa: F401
except ImportError:
    from distutils.core import Extension, setup  # noqa: F401

NAME = "bytecheckpoint"
VERSION = "0.0.2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIRES = [
    "torch>=2.1.0,<=2.5.0",
    "mmh3>=4.1.0,<=4.2.0",
    "universal_pathlib>=0.2.2,<=1.0.0",
    "cachetools>=5.4.0,<=10.0.0",
    "safetensors>=0.4.5,<=1.0.0",
]
try:
    version_file_path = "bytecheckpoint/version.py"
    with open(version_file_path) as f:
        (version,) = re.findall('__version__ = "(.*)"', f.read())
    VERSION = version

except:  # noqa: E722
    raise ValueError(f"Failed to get version from {version_file_path}.")


def package_files(directory, f=None):
    if isinstance(directory, (list, tuple)):
        l = [package_files(d, f=f) for d in directory]
        return [item for sublist in l for item in sublist]
    directory = os.path.join(BASE_DIR, directory)
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            if callable(f):
                if f(filename):
                    paths.append(os.path.join("..", path, filename))
                continue
            if isinstance(f, str):
                if re.match(f, filename):
                    paths.append(os.path.join("..", path, filename))
                continue
            paths.append(os.path.join("..", path, filename))
    # print(paths)
    return paths


with open("requirements.txt") as f_req:
    required = []
    for line in f_req:
        if line.startswith("-"):
            continue
        line = line.strip()
        if "-" in line:
            line = line[line.index("-")]
        required.append(line)

# Append the required packages to the list.
REQUIRES += required

# Add description.
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=NAME,
    version=VERSION,
    description="ByteCheckpoint: An Unified Checkpointing Library for LFMs",
    author="ByteDance-Seed-MLSys",
    license="Apache 2.0",
    author_email="wanborui@connect.hku.hk, mingjihan@bytedance.com",
    url="https://github.com/ByteDance-Seed/ByteCheckpoint.git",
    packages=["bytecheckpoint"],
    package_data={
        "bytecheckpoint": package_files(["bytecheckpoint"]),
    },
    install_requires=REQUIRES,
    python_requires=">=3.8",
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)

#!/usr/bin/env python
"""
in order to create virtual environment with the required (dev) dependencies:

    make venv

TODO use bandit to check for vulnerabilities:

conda install bandit
bandit ./multivar_horner/*.py

"""

import os
import re
import sys
from os.path import abspath, isfile, join, pardir

PACKAGE = "multivar_horner"
VERSION_FILE = "VERSION"
VIRT_ENV_NAME = PACKAGE
VIRT_ENV_COMMAND = f". ~/miniconda3/etc/profile.d/conda.sh; conda activate {VIRT_ENV_NAME}; "
PY_VERSION_IDS = [
    "37",
    "38",
    "39",
]  # the supported python versions to create wheels for
PYTHON_TAG = ".".join([f"py{v}" for v in PY_VERSION_IDS])


# TODO not required, set version in version file
def get_version():
    return open(VERSION_FILE, "r").read().strip()


def parse_version(new_version_input="", old_version_str="1.0.0"):
    new_version_input = re.search(r"\d\.\d\.\d+", new_version_input)

    if new_version_input is None:
        raise ValueError  # will cause new input request
    else:
        new_version_input = new_version_input.group()

    # print(new_version_input)

    split_new_version = [int(x) for x in new_version_input.split(".")]
    # print(split_new_version)
    split_old_version = [int(x) for x in old_version_str.split(".")]
    # print(split_old_version)

    for i in range(3):
        if split_new_version[i] > split_old_version[i]:
            break
        if split_new_version[i] < split_old_version[i]:
            raise ValueError  # will cause new input request

    return new_version_input


def set_version(new_version_str):
    with open(VERSION_FILE, "w") as version_file:
        version_file.write(new_version_str)


def routine(cmd=None, message="", option1="next", option2="exit"):
    while 1:
        print(message)

        if cmd:
            print("running command:", cmd)
            os.system(cmd)

        print("__________\nDone. Options:")
        print("1)", option1)
        print("2)", option2)
        print("anything else to repeat this step.")
        try:
            inp = int(input())

            if inp == 1:
                print("==============")
                break
            if inp == 2:
                sys.exit()

        except ValueError:
            pass
        print("================")


if __name__ == "__main__":

    print('Do you want to switch to the "dev" branch? Commit before switching branch!')
    print("1) yes, change now.")
    print("2) no, exit")
    print("anything else skip.")
    try:
        inp = int(input())
        if inp == 1:
            os.system("git checkout dev")
            print("==============")
        if inp == 2:
            sys.exit()
    except ValueError:
        pass

    old_version = get_version()

    print("The actual version number is:", old_version)
    print("Enter new version number:")
    version_input = None
    while 1:
        try:
            version_input = input()
            version_str = parse_version(version_input, old_version)
            set_version(version_str)
            break
        except ValueError:
            print(
                f'Invalid version input. Should be of format "x.x.xxx" and higher than the old version {old_version}.'
            )
            pass  # try again

    version = get_version()
    print("the version number has been set to:", version)
    print("=====================")

    routine(
        None,
        "Remember to properly specify all supported python versions in publish.py and setup.py",
    )
    routine(
        None,
        "Maybe re-pin the test dependencies (requirements.txt) with pip-compile!"
        " Commands are written in the beginning of this script",
    )
    routine(
        None,
        "Have all pinned dependencies been listed in setup.py and the Documentation?",
    )
    routine(None, "Have all (new) features been documented?")
    routine(None, f"Remember to write a changelog for version {version}")
    routine(None, "Remember to update MANIFEST.in if the packate content has chanted")

    print("___________")
    print("Running TESTS:")

    # TODO ask
    # routine(VIRT_ENV_COMMAND + "pip-compile requirements_numba.in;pip-sync",
    #      'pinning the requirements.txt and bringing virtualEnv to exactly the specified state:', 'next: build check')

    routine(
        f"{VIRT_ENV_COMMAND} rstcheck *.rst",
        "checking syntax of all .rst files:",
        "next: build check",
    )

    print("generating documentation now...")
    os.system("(cd ./docs && exec make html)")
    print("done.")

    # IMPORTANT: -r flag to rebuild tox virtual env
    # only when dependencies have changed!
    rebuild_flag = ""
    print("when the dependencies (in requirements_tests.txt) have changed enter 1 (-> rebuild tox)")
    try:
        inp = int(input())
        if inp == 1:
            rebuild_flag = "-r"
    except ValueError:
        pass

    routine(f"{VIRT_ENV_COMMAND} tox {rebuild_flag} -e py37", "run tests")
    print("Tests finished.")

    routine(
        None,
        "Please commit your changes, push and wait if Travis tests build successfully. "
        "Only then merge them into the master.",
        "CI tests passed & merge into master complete. Build and upload now.",
    )

    # TODO do this automatically, problem are the commit messages <- often the same as changelog!
    # git commit --message
    # git push dev

    # TODO wait for Travis to finish positively

    # if not in master

    # TODO ask to push in master
    # git merge ...

    # TODO switching to master

    print("=================")
    print("PUBLISHING:")

    # routine("python3 setup.py sdist bdist_wheel upload", 'Uploading the package now.') # deprecated
    # new twine publishing routine:
    # https://packaging.python.org/tutorials/packaging-projects/
    # delete the build folder before to get a fresh build
    routine(
        f"rm -r -f build; python setup.py sdist bdist_wheel --python-tag {PYTHON_TAG}",
        "building the package now.",
        "build done. check the included files! test uploading.",
    )

    path = abspath(join(__file__, pardir, "dist"))
    all_archives_this_version = [f for f in os.listdir(path) if isfile(join(path, f)) and version_str in f]
    paths2archives = [abspath(join(path, f)) for f in all_archives_this_version]
    command = "twine upload --repository-url https://test.pypi.org/legacy/ " + " ".join(paths2archives)

    # upload all archives of this version
    routine(VIRT_ENV_COMMAND + command, "testing if upload works.")

    command = "twine upload " + " ".join(paths2archives)
    routine(VIRT_ENV_COMMAND + command, "real upload to PyPI.")

    # tag erstellen
    routine(None, "Do you want to create a git release tag?", "Yes", "No")
    routine(f"git tag -a v{version} -m 'Version {version}'; git push --tags", "Creating tag")
    print(f"______________\nCongrats! Published version {version}.")

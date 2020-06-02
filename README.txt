Preliminaries
=============

This solver constsists of two parts
1) Sat-based solving (the corresponding code is written in python, using Glucose4 solver of python-sat)
2) Dp-based solving (written in c++)

It's possible to separate them to separate solvers given minor code/compilation flags changes.
(More precisely, the c++ solver is self-sufficient if built with -DAS_MAIN,
while python solver is self-sufficient if you remove line "transfer_to_cpp(instance, VC)")

Preparations
============

In case it's possible, it's much simpler to setup additional software using package managers.

The requirements are:

0. Reasonably modern enough g++ (apparently c++14 is enough)
1. Reasonably modern python3 (was tested with python3.7.7 & python3.8rc1),
however if you strip away all type annotations, much lower version should suffice
1. pip3 install python-sat
2. pip3 install cffi

Building
========

The DP-solver should be available as a shared library:

$ g++ -Wall -Wextra code.cpp  -o cppsolve.so -std=c++14 -shared  -fPIC -O2 -g

Running
=======

$ ./solve.py < example

(Be careful that c++ solver allocates a lot of memory though)

Building portable TGZ
=====================

This section describes how the TGZ for optil submission was built.

$ mkdir env && cd env

# xenial is optil's ubuntu release
# you might want to choose another mirror
$ [sudo] debootstrap xenial . https://mirror.yandex.ru/ubuntu/

Before the next commands run "chroot env"

edit /etc/apt/sources.list so it is roughly as follows:
"deb https://mirror.yandex.ru/ubuntu xenial-updates main universe"
"deb https://mirror.yandex.ru/ubuntu xenial main universe"

$ apt install software-properties-common # apparently needed for ppa to work
$ add-apt-repository ppa:deadsnakes/ppa # more modern python
$ apt update && apt install python3.7 python3.7-dev python3-pip

$ # should make "python3 --version" be "Python3.7.7", more simple way would
$ # have been to simply copy /usr/bin/python3.7 to /usr/bin/python3
$ update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 0

$ apt install zlib1g-dev g++ python3-pip # zlib1g needed for python-sat

Finally install dependencies
$ pip3 install --upgrade pip
$ pip3 install python-sat cffi pyinstaller

OK! We finally ready to finish the build, return to the host system

Compile cppsolver as before:

$ g++ -Wall -Wextra code.cpp  -o cppsolve.so -std=c++14 -shared  -fPIC -O2 -g

Build python solver:

$ [sudo] cp solve.py env/root/solve.py
$ [sudo] sudo chroot env bash -c "cd && pyinstaller solve.py --noconfirm"

Join solvers together & pack the tar.gz
$ [sudo] cp cppsolve.so env/root/dist/solve
$ [sudo] tar -czvf solve.tgz -C env/root/dist/solve .

Voila!

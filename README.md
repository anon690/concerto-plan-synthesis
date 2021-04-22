This repository contains the implementation and experiments for the
paper *SMT-Based Planning Synthesis for Distributed System
Reconfigurations*.

# Required packages

This program depends on Z3 and its Python API, which are available to
install through most package managers, e.g., by running

`sudo apt-get python3-z3`

# Running benchmarks

To run the examples without installing, you can temporarily extend the
python path. From this directory, run

`export PYTHONPATH=$PYTHONPATH:$PWD`

and run the examples using python. The `open_stack.py` example has no
parameter. For the other examples, you must specify the number of
components in the assembly, e.g.:

`python concerto_scheduling_examples/linear_deployment.py 10`

# Organisation of this repository

* `concerto` contains the implementation of the synthesis method
  * `assembly.py` includes classes used to describe Concerto components and assemblies
  * `execution.py` includes methods related to executing Concerto reconfigurations
  * `scheduling.py` includes the synthesis algorithm and related classes
  * `fo_logic.py` includes classes to describre first-order expressions used in the SMT encoding of the scheduling problems
* `concerto_scheduling_examples` contains the benchmarks described in the paper
* `data` contains the experimental data obtained by running those benchmarks
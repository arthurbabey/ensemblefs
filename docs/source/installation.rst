Installation
============

How to Install ensemblefs
---------------------------------------


You can build and install the package from source, you can follow these steps. First, clone the repository and navigate to the project root directory:

.. code-block:: bash

    git clone https://github.com/arthurbabey/ensemblefs.git
    cd ensemblefs

Then, install the package using `pip`:

.. code-block:: bash

    pip install .

This will build and install the package locally in your environment.

If you prefer to install the package in **editable** mode (allowing code changes to take effect without reinstalling), you can use:

.. code-block:: bash

    pip install -e .

If you want to help in the development of the package and run the tests, you can install the package with the development dependencies:

.. code-block:: bash

    pip install -e .[dev]

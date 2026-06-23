Installation
=============

Requirements
------------

- Python 3.12 or later
- pip or `uv <https://docs.astral.sh/uv/>`_

Basic Installation
------------------

Install the package from source:

.. code-block:: bash

    pip install -e .

Or using uv:

.. code-block:: bash

    uv pip install -e .

Optional Dependencies
---------------------

For development and testing:

.. code-block:: bash

    pip install -e ".[dev]"

For experiment tracking with Weights & Biases:

.. code-block:: bash

    pip install -e ".[tracking]"

Verify Installation
-------------------

To verify the installation was successful:

.. code-block:: python

    import sda_mc
    print(sda_mc.__version__)

Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

    pip install -r docs/requirements.txt
    cd docs && make html

The built documentation will be in ``docs/_build/html/``.

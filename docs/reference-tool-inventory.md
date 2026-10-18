# Dependency-stack survey

Surveyed 2026-10-17. This anonymized catalog describes an external example stack, not the installed environment of Payment Risk Workbench. See this project's `pyproject.toml` and `poetry.lock` for its actual dependencies.

## What the scan establishes

- 150 locked package entries; 148 unique package names.
- 18 directly declared package dependencies, excluding Python itself.
- This is an inventory of the surveyed stack, not a dependency installation plan for Payment Risk Workbench.
- Versions below are the surveyed versions, not independently verified compatibility recommendations.
- A lockfile includes alternatives and conditional dependencies. The active platform, Python version, requested groups, extras and dependency constraints determine the installed set.
- `numba` and `llvmlite` each appear at two versions. Do not install both versions into one environment. This scan does not claim to have resolved or tested the surveyed environment.
- The parent column reports dependency declarations found in the lockfile, including conditional declarations; it does not assert every edge is active on this Mac.
- Package descriptions below come from lockfile metadata. An installed package does not necessarily represent a tool a project uses directly.

## Important distinctions

- `mlflow → aiohttp → aiohappyeyeballs`: asynchronous HTTP networking pulls in connection-handling support.
- `databricks-sdk` is pulled in by MLflow components; its presence does not establish a Databricks deployment.
- `docker` here is the Python Docker SDK. The Docker engine/CLI used for containers is a separate installation.
- OpenTelemetry packages, GraphQL packages and additional web servers are supporting dependencies; their presence alone does not establish separately configured tracing, GraphQL or serving systems.
- `pywin32` and `appnope` support different operating systems. An NVIDIA runtime entry does not mean this project requires an NVIDIA GPU.
- Poetry, poetry-core, pipx, system libraries, browser executables and cloud services are not exhaustively represented by this application lockfile. Those requirements must be inventoried separately from application dependencies.

## Complete lockfile inventory

“Direct” identifies the group in `pyproject.toml`. “Transitive” means another package declares the dependency. Direct imports such as NumPy, SciPy, Matplotlib, joblib and Pydantic can still be transitive in the surveyed configuration. Our project should declare the libraries it imports.

| Package | Locked version | Declaration | Declared parent packages | Description from lockfile |
|---|---|---|---|---|
| `aiohappyeyeballs` | `2.7.1` | Transitive | `aiohttp` | Happy Eyeballs for asyncio |
| `aiohttp` | `3.14.3` | Transitive | `mlflow` | Async http client/server framework (asyncio) |
| `aiosignal` | `1.4.0` | Transitive | `aiohttp` | aiosignal: a list of registered asynchronous callbacks |
| `alembic` | `1.20.0` | Transitive | `mlflow`, `optuna` | A database migration tool for SQLAlchemy. |
| `annotated-doc` | `0.0.5` | Transitive | `fastapi` | Document parameters, class attributes, return types, and variables inline, with Annotated. |
| `annotated-types` | `0.8.0` | Transitive | `pydantic` | Reusable constraint types to use with typing.Annotated |
| `anyio` | `4.15.1` | Transitive | `httpx`, `mlflow-skinny`, `starlette` | High-level concurrency and networking framework on top of asyncio or Trio |
| `appnope` | `1.0.0` | Transitive | `ipykernel` | Disable App Nap on macOS >= 10.9 |
| `asttokens` | `3.0.2` | Transitive | `stack-data` | Annotate AST trees with source code positions |
| `attrs` | `26.1.0` | Transitive | `aiohttp`, `jsonschema`, `referencing` | Classes Without Boilerplate |
| `blinker` | `1.9.0` | Transitive | `flask` | Fast, simple object-to-object and broadcast signaling |
| `cachetools` | `7.2.0` | Transitive | `mlflow-skinny`, `mlflow-tracing` | Extensible memoizing collections and decorators |
| `catboost` | `1.2.10` | Direct: experimentation | — | CatBoost Python Package |
| `certifi` | `2026.7.22` | Transitive | `httpcore`, `httpx`, `requests` | Python package for providing Mozilla's CA Bundle. |
| `cffi` | `2.1.1` | Transitive | `cryptography`, `pyzmq` | Foreign Function Interface for Python calling C code. |
| `charset-normalizer` | `3.5.1` | Transitive | `requests` | The Real First Universal Charset Detector. Open, modern and actively maintained alternative to Chardet. |
| `choreographer` | `1.4.0` | Transitive | `kaleido` | Devtools Protocol implementation for chrome. |
| `click` | `8.5.0` | Transitive | `flask`, `mlflow-skinny`, `uvicorn` | Composable command line interface toolkit |
| `cloudpickle` | `3.1.2` | Transitive | `joblib`, `mlflow-skinny`, `shap` | Pickler class to extend the standard pickle.Pickler functionality |
| `colorama` | `0.4.6` | Transitive | `colorlog`, `ipython`, `pytest`, `tqdm` | Cross-platform colored terminal text. |
| `colorlog` | `6.12.0` | Transitive | `optuna` | Add colours to the output of Python's logging module. |
| `comm` | `0.2.3` | Transitive | `ipykernel` | Jupyter Python Comm implementation, for usage in ipykernel, xeus-python etc. |
| `contourpy` | `1.4.0` | Transitive | `matplotlib` | Python library for calculating contours of 2D quadrilateral grids |
| `coverage` | `7.16.1` | Transitive | `pytest-cov` | Code coverage measurement for Python |
| `cryptography` | `50.0.1` | Transitive | `google-auth`, `mlflow` | cryptography is a package which provides cryptographic recipes and primitives to Python developers. |
| `cycler` | `0.12.1` | Transitive | `matplotlib` | Composable style cycles |
| `databricks-sdk` | `0.141.0` | Transitive | `mlflow-skinny`, `mlflow-tracing` | Databricks SDK for Python (Beta) |
| `debugpy` | `1.8.22` | Transitive | `ipykernel` | An implementation of the Debug Adapter Protocol for Python |
| `docker` | `7.2.0` | Transitive | `mlflow` | A Python library for the Docker Engine API. |
| `executing` | `2.2.1` | Transitive | `stack-data` | Get the currently executing AST node of a frame, and other information |
| `fastapi` | `0.141.1` | Direct: production | `mlflow-skinny` | FastAPI framework, high performance, easy to learn, fast to code, ready for production |
| `fastjsonschema` | `2.22.2` | Transitive | `nbformat` | Fastest Python implementation of JSON schema |
| `flask` | `3.1.3` | Transitive | `flask-cors`, `mlflow` | A simple framework for building complex web applications. |
| `flask-cors` | `6.0.5` | Transitive | `mlflow` | A Flask extension simplifying CORS support |
| `fonttools` | `4.66.0` | Transitive | `matplotlib` | Tools to manipulate font files |
| `frozenlist` | `1.8.0` | Transitive | `aiohttp`, `aiosignal` | A list-like structure which implements collections.abc.MutableSequence |
| `gitdb` | `4.0.12` | Transitive | `gitpython` | Git Object Database |
| `gitpython` | `3.1.62` | Transitive | `mlflow-skinny` | GitPython is a Python library used to interact with Git repositories |
| `google-auth` | `2.58.1` | Transitive | `databricks-sdk` | Google Authentication Library |
| `graphene` | `3.4.3` | Transitive | `mlflow` | GraphQL Framework for Python |
| `graphql-core` | `3.2.12` | Transitive | `graphene`, `graphql-relay` | GraphQL implementation for Python, a port of GraphQL.js, the JavaScript reference implementation for GraphQL. |
| `graphql-relay` | `3.2.0` | Transitive | `graphene` | Relay library for graphql-core |
| `graphviz` | `0.21` | Transitive | `catboost` | Simple Python interface for Graphviz |
| `gunicorn` | `26.2.0` | Transitive | `mlflow` | WSGI HTTP Server for UNIX |
| `h11` | `0.16.0` | Transitive | `httpcore`, `uvicorn` | A pure-Python, bring-your-own-I/O implementation of HTTP/1.1 |
| `httpcore` | `1.0.9` | Transitive | `httpx` | A minimal low-level HTTP client. |
| `httpx` | `0.28.1` | Direct: dev | — | The next generation HTTP client. |
| `huey` | `3.4.0` | Transitive | `mlflow` | a little task queue |
| `idna` | `3.20` | Transitive | `anyio`, `httpx`, `requests`, `yarl` | Internationalized Domain Names in Applications (IDNA) |
| `importlib-metadata` | `9.0.1` | Transitive | `mlflow-skinny` | Read metadata from Python packages |
| `iniconfig` | `2.3.0` | Transitive | `pytest` | brain-dead simple config-ini parsing |
| `ipykernel` | `7.3.0` | Direct: dev | — | IPython Kernel for Jupyter |
| `ipython` | `9.17.1` | Transitive | `ipykernel` | IPython: Productive Interactive Computing |
| `ipython-pygments-lexers` | `1.1.1` | Transitive | `ipython` | Defines a variety of Pygments lexers for highlighting IPython code. |
| `itsdangerous` | `2.2.0` | Transitive | `flask` | Safely pass data to untrusted environments and back. |
| `jedi` | `0.20.0` | Transitive | `ipython` | An autocompletion tool for Python that can be used for text editors. |
| `jinja2` | `3.1.6` | Transitive | `flask` | A very fast and expressive template engine. |
| `joblib` | `1.6.0` | Transitive | `scikit-learn` | Lightweight pipelining with Python functions |
| `jsonschema` | `4.26.0` | Transitive | `nbformat` | An implementation of JSON Schema validation for Python |
| `jsonschema-specifications` | `2025.9.1` | Transitive | `jsonschema` | The JSON Schema meta-schemas and vocabularies, exposed as a Registry |
| `jupyter-client` | `8.10.0` | Transitive | `ipykernel` | Jupyter protocol implementation and client libraries |
| `jupyter-core` | `5.9.1` | Transitive | `ipykernel`, `jupyter-client`, `nbformat` | Jupyter core package. A base package on which Jupyter projects rely. |
| `kaleido` | `1.4.0` | Direct: experimentation | — | Plotly graph export library |
| `kiwisolver` | `1.5.1` | Transitive | `matplotlib` | A fast implementation of the Cassowary constraint solver |
| `lightgbm` | `4.7.0` | Direct: production | — | LightGBM Python-package |
| `llvmlite` | `0.34.0` | Transitive | `numba`, `shap` | lightweight wrapper around basic LLVM functionality |
| `llvmlite` | `0.49.0` | Transitive | `numba`, `shap` | lightweight wrapper around basic LLVM functionality |
| `logistro` | `2.0.1` | Transitive | `choreographer`, `kaleido` | Simple wrapper over logging for a couple basic features |
| `mako` | `1.4.3` | Transitive | `alembic` | A super-fast templating language that borrows the best ideas from the existing templating languages. |
| `markupsafe` | `3.0.3` | Transitive | `flask`, `jinja2`, `mako`, `werkzeug` | Safely add untrusted strings to HTML/XML markup. |
| `matplotlib` | `3.11.2` | Transitive | `catboost`, `mlflow` | Python plotting package |
| `matplotlib-inline` | `0.2.2` | Transitive | `ipykernel`, `ipython` | Inline Matplotlib backend for Jupyter |
| `mlflow` | `3.16.1` | Direct: experimentation | — | MLflow is an open source platform for the complete machine learning lifecycle |
| `mlflow-skinny` | `3.16.1` | Transitive | `mlflow` | MLflow is an open source platform for the complete machine learning lifecycle |
| `mlflow-tracing` | `3.16.1` | Transitive | `mlflow` | MLflow Tracing SDK is an open-source, lightweight Python package that only includes the minimum set of dependencies and functionality to instrument your code/models/agents with MLflow Tracing. |
| `multidict` | `6.9.1` | Transitive | `aiohttp`, `yarl` | multidict implementation |
| `narwhals` | `2.26.0` | Transitive | `lightgbm`, `plotly`, `scikit-learn` | Extremely lightweight compatibility layer between dataframe libraries |
| `nbformat` | `5.11.1` | Direct: experimentation | — | The Jupyter Notebook format |
| `nest-asyncio2` | `1.7.3` | Transitive | `ipykernel` | Patch asyncio to allow nested event loops |
| `numba` | `0.51.2` | Transitive | `shap` | compiling Python code using LLVM |
| `numba` | `0.67.0` | Transitive | `shap` | compiling Python code using LLVM |
| `numpy` | `2.5.3` | Transitive | `catboost`, `contourpy`, `lightgbm`, `matplotlib`, `mlflow`, `numba`, `optuna`, `pandas`, `scikit-learn`, `scipy`, `shap`, `skops`, `xgboost` | Fundamental package for array computing in Python |
| `nvidia-nccl-cu13` | `2.32.3` | Transitive | `xgboost` | NVIDIA Collective Communication Library (NCCL) Runtime |
| `opentelemetry-api` | `1.44.0` | Transitive | `mlflow-skinny`, `mlflow-tracing`, `opentelemetry-sdk`, `opentelemetry-semantic-conventions` | OpenTelemetry Python API |
| `opentelemetry-proto` | `1.44.0` | Transitive | `mlflow-skinny`, `mlflow-tracing` | OpenTelemetry Python Proto |
| `opentelemetry-sdk` | `1.44.0` | Transitive | `mlflow-skinny`, `mlflow-tracing` | OpenTelemetry Python SDK |
| `opentelemetry-semantic-conventions` | `0.65b0` | Transitive | `opentelemetry-sdk` | OpenTelemetry Semantic Conventions |
| `optuna` | `5.0.0` | Direct: experimentation | — | A hyperparameter optimization framework |
| `orjson` | `3.12.0` | Transitive | `kaleido` | Fast, correct Python JSON library supporting dataclasses, datetimes, and numpy |
| `packaging` | `26.3` | Transitive | `ipykernel`, `kaleido`, `matplotlib`, `mlflow-skinny`, `mlflow-tracing`, `optuna`, `plotly`, `pytest`, `shap`, `skops` | Core utilities for Python packages |
| `pandas` | `3.0.6` | Direct: production | `catboost`, `mlflow`, `shap` | Powerful data structures for data analysis, time series, and statistics |
| `parso` | `0.8.7` | Transitive | `jedi` | A Python Parser |
| `pexpect` | `4.9.0` | Transitive | `ipython` | Pexpect allows easy control of interactive console applications. |
| `pillow` | `12.3.0` | Transitive | `matplotlib` | Python Imaging Library (fork) |
| `platformdirs` | `4.11.12` | Transitive | `choreographer`, `jupyter-core` | A small Python package for determining appropriate platform-specific dirs, e.g. a `user data dir`. |
| `plotly` | `7.1.0` | Direct: experimentation | `catboost` | An open-source interactive data visualization library for Python |
| `pluggy` | `1.6.0` | Transitive | `pytest`, `pytest-cov` | plugin and hook calling mechanisms for python |
| `prettytable` | `3.18.0` | Transitive | `skops` | A simple Python library for easily displaying tabular data in a visually appealing ASCII table format |
| `prompt-toolkit` | `3.0.53` | Transitive | `ipython` | Library for building powerful interactive command lines in Python |
| `propcache` | `0.5.4` | Transitive | `aiohttp`, `yarl` | Accelerated property cache |
| `protobuf` | `6.33.6` | Transitive | `databricks-sdk`, `mlflow-skinny`, `mlflow-tracing`, `opentelemetry-proto` | No description recorded. |
| `psutil` | `7.2.2` | Transitive | `ipykernel`, `ipython` | Cross-platform lib for process and system monitoring. |
| `ptyprocess` | `0.7.0` | Transitive | `pexpect` | Run a subprocess in a pseudo terminal |
| `pure-eval` | `0.2.4` | Transitive | `stack-data` | Safely evaluate AST nodes without side effects |
| `pyarrow` | `25.0.1` | Direct: production | `mlflow` | Python library for Apache Arrow |
| `pyasn1` | `0.6.4` | Transitive | `pyasn1-modules` | Pure-Python implementation of ASN.1 types and DER/BER/CER codecs (X.208) |
| `pyasn1-modules` | `0.4.2` | Transitive | `google-auth` | A collection of ASN.1-based protocols modules |
| `pycparser` | `3.0` | Transitive | `cffi` | C parser in Python |
| `pydantic` | `2.13.5` | Transitive | `fastapi`, `mlflow-skinny`, `mlflow-tracing` | Data validation using Python type hints |
| `pydantic-core` | `2.46.5` | Transitive | `pydantic` | Core functionality for Pydantic validation and serialization |
| `pygments` | `2.21.0` | Transitive | `ipython`, `ipython-pygments-lexers`, `pytest` | Pygments is a syntax highlighting package written in Python. |
| `pyparsing` | `3.3.3` | Transitive | `matplotlib` | pyparsing - Classes and methods to define and execute parsing grammars |
| `pytest` | `9.1.1` | Direct: dev | `pytest-cov` | pytest: simple powerful testing with Python |
| `pytest-cov` | `7.1.0` | Direct: dev | — | Pytest plugin for measuring coverage. |
| `python-dateutil` | `2.9.0.post0` | Transitive | `graphene`, `jupyter-client`, `matplotlib`, `pandas` | Extensions to the standard Python datetime module |
| `python-dotenv` | `1.2.3` | Transitive | `mlflow-skinny` | Read key-value pairs from a .env file and set them as environment variables |
| `pywin32` | `312` | Transitive | `docker` | Python for Windows Extensions |
| `pyyaml` | `6.0.3` | Transitive | `mlflow-skinny`, `optuna` | YAML parser and emitter for Python |
| `pyzmq` | `27.2.0` | Transitive | `ipykernel`, `jupyter-client` | Python bindings for 0MQ |
| `referencing` | `0.37.0` | Transitive | `jsonschema`, `jsonschema-specifications` | JSON Referencing + Python |
| `requests` | `2.34.2` | Transitive | `databricks-sdk`, `docker`, `mlflow-skinny` | Python HTTP for Humans. |
| `rpds-py` | `2026.6.3` | Transitive | `jsonschema`, `referencing` | Python bindings to Rust's persistent data structures (rpds) |
| `scikit-learn` | `1.9.1` | Direct: production | `mlflow`, `shap`, `skops` | A set of python modules for machine learning and data mining |
| `scipy` | `1.18.1` | Transitive | `catboost`, `lightgbm`, `mlflow`, `scikit-learn`, `shap`, `skops`, `xgboost` | Fundamental algorithms for scientific computing in Python |
| `setuptools` | `84.0.0` | Transitive | `numba` | Most extensible Python build backend with support for C/C++ extension modules |
| `shap` | `0.52.0` | Direct: experimentation | — | A unified approach to explain the output of any machine learning model. |
| `simplejson` | `4.1.2` | Transitive | `choreographer` | Simple, fast, extensible JSON encoder/decoder for Python |
| `six` | `1.17.0` | Transitive | `catboost`, `python-dateutil` | Python 2 and 3 compatibility utilities |
| `skops` | `0.16.0` | Transitive | `mlflow` | A set of tools, related to machine learning in production. |
| `slicer` | `0.0.8` | Transitive | `shap` | A small package for big slicing. |
| `smmap` | `5.0.3` | Transitive | `gitdb` | A pure Python implementation of a sliding window memory map manager |
| `sqlalchemy` | `2.1.0` | Transitive | `alembic`, `mlflow`, `optuna` | Database Abstraction Library |
| `sqlparse` | `0.6.0` | Transitive | `mlflow-skinny` | A non-validating SQL parser. |
| `stack-data` | `0.6.3` | Transitive | `ipython` | Extract data from python stack frames and tracebacks for informative displays |
| `starlette` | `1.7.0` | Transitive | `fastapi`, `mlflow-skinny` | The little ASGI library that shines. |
| `threadpoolctl` | `3.7.0` | Transitive | `scikit-learn` | threadpoolctl |
| `tornado` | `6.5.10` | Transitive | `ipykernel`, `jupyter-client` | Tornado is a Python web framework and asynchronous networking library, originally developed at FriendFeed. |
| `tqdm` | `4.70.1` | Transitive | `optuna`, `shap` | Fast, Extensible Progress Meter |
| `traitlets` | `5.16.1` | Transitive | `ipykernel`, `ipython`, `jupyter-client`, `jupyter-core`, `matplotlib-inline`, `nbformat` | Traitlets Python configuration system |
| `typing-extensions` | `4.16.0` | Transitive | `aiohttp`, `aiosignal`, `alembic`, `anyio`, `fastapi`, `graphene`, `jupyter-client`, `mlflow-skinny`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-semantic-conventions`, `pydantic`, `pydantic-core`, `referencing`, `sqlalchemy`, `starlette`, `typing-inspection` | Backported and Experimental Type Hints for Python 3.9+ |
| `typing-inspection` | `0.4.4` | Transitive | `fastapi`, `pydantic` | Runtime typing introspection tools |
| `tzdata` | `2026.4` | Transitive | `pandas` | Provider of IANA time zone data |
| `urllib3` | `2.8.0` | Transitive | `databricks-sdk`, `docker`, `requests` | HTTP library with thread-safe connection pooling, file post, and more. |
| `uvicorn` | `0.54.0` | Direct: production | `mlflow-skinny` | The lightning-fast ASGI server. |
| `waitress` | `3.0.2` | Transitive | `mlflow` | Waitress WSGI server |
| `wcwidth` | `0.9.1` | Transitive | `prettytable`, `prompt-toolkit` | Measures the displayed width of unicode strings in a terminal |
| `werkzeug` | `3.1.8` | Transitive | `flask`, `flask-cors` | The comprehensive WSGI web application library. |
| `xgboost` | `3.4.1` | Direct: experimentation | — | XGBoost Python Package |
| `yarl` | `1.25.1` | Transitive | `aiohttp` | Yet another URL library |
| `zipp` | `4.1.0` | Transitive | `importlib-metadata` | Backport of pathlib-compatible object wrapper for zip files |

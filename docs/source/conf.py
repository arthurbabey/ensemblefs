project = "ensemblefs"
copyright = "2024, Arthur Babey"
author = "Arthur Babey"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google or NumPy style docstrings
    "sphinx.ext.viewcode",  # Optional, adds links to highlighted source code
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "inherited-members": True,
    "show-inheritance": True,
}

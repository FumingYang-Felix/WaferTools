"""Stub replacement for Intel's sklearnex when package isn't available.
Provides a minimal shim that re-exports scikit-learn sub-modules used by msemalign (neighbors, decomposition, etc.).
"""
import importlib, sys, types

_sklearn = importlib.import_module('sklearn')

# Create stub module
_module = types.ModuleType('sklearnex')

# Re-export key subpackages by forwarding to sklearn equivalents
for sub in ['neighbors', 'decomposition', 'cluster', 'linear_model', 'svm', 'metrics']:
    try:
        setattr(_module, sub, importlib.import_module(f'sklearn.{sub}'))
    except ImportError:
        pass
    else:
        # register as sklearnex.submodule so import finds it
        sys.modules[f'sklearnex.{sub}'] = getattr(_module, sub)

# Make sklearnex patch visible to future imports
sys.modules['sklearnex'] = _module 
Provides a minimal shim that re-exports scikit-learn sub-modules used by msemalign (neighbors, decomposition, etc.).
"""
import importlib, sys, types

_sklearn = importlib.import_module('sklearn')

# Create stub module
_module = types.ModuleType('sklearnex')

# Re-export key subpackages by forwarding to sklearn equivalents
for sub in ['neighbors', 'decomposition', 'cluster', 'linear_model', 'svm', 'metrics']:
    try:
        setattr(_module, sub, importlib.import_module(f'sklearn.{sub}'))
    except ImportError:
        pass
    else:
        # register as sklearnex.submodule so import finds it
        sys.modules[f'sklearnex.{sub}'] = getattr(_module, sub)

# Make sklearnex patch visible to future imports
sys.modules['sklearnex'] = _module 
 
Provides a minimal shim that re-exports scikit-learn sub-modules used by msemalign (neighbors, decomposition, etc.).
"""
import importlib, sys, types

_sklearn = importlib.import_module('sklearn')

# Create stub module
_module = types.ModuleType('sklearnex')

# Re-export key subpackages by forwarding to sklearn equivalents
for sub in ['neighbors', 'decomposition', 'cluster', 'linear_model', 'svm', 'metrics']:
    try:
        setattr(_module, sub, importlib.import_module(f'sklearn.{sub}'))
    except ImportError:
        pass
    else:
        # register as sklearnex.submodule so import finds it
        sys.modules[f'sklearnex.{sub}'] = getattr(_module, sub)

# Make sklearnex patch visible to future imports
sys.modules['sklearnex'] = _module 
Provides a minimal shim that re-exports scikit-learn sub-modules used by msemalign (neighbors, decomposition, etc.).
"""
import importlib, sys, types

_sklearn = importlib.import_module('sklearn')

# Create stub module
_module = types.ModuleType('sklearnex')

# Re-export key subpackages by forwarding to sklearn equivalents
for sub in ['neighbors', 'decomposition', 'cluster', 'linear_model', 'svm', 'metrics']:
    try:
        setattr(_module, sub, importlib.import_module(f'sklearn.{sub}'))
    except ImportError:
        pass
    else:
        # register as sklearnex.submodule so import finds it
        sys.modules[f'sklearnex.{sub}'] = getattr(_module, sub)

# Make sklearnex patch visible to future imports
sys.modules['sklearnex'] = _module 
 
 
 
 
 
 
 
 
 
 
 
 
 
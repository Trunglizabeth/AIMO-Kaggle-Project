import importlib
try:
    importlib.import_module('src.engine_api')
    print('IMPORT_OK')
except Exception as e:
    print('IMPORT_ERROR', type(e).__name__, str(e))
    raise

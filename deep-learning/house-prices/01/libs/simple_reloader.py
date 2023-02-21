def reload(module):
    import sys, imp
    imp.reload(sys.modules[module])
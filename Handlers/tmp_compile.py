from pathlib import Path
import py_compile
p = Path('app.py')
try:
    py_compile.compile(str(p), doraise=True)
    print('ok')
except py_compile.PyCompileError as e:
    print('ERROR')
    print(e.msg)
    raise

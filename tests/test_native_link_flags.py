"""Regression: runtime directories must not become compiler input filenames."""
import ast
from pathlib import Path,PurePosixPath
import shlex
import types
import unittest


class NativeLinkFlags(unittest.TestCase):
    def test_only_option_bound_directory_arguments(self):
        path=Path(__file__).resolve().parents[1]/'noposplat_jittor/rasterizer.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='options')
        for base in ('/tmp/nps','/tmp/nps with spaces'):
            root=PurePosixPath(base)
            namespace={'ROOT':root,'LIB':root/'native/build/libnps_raster.so'}
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
            output=types.SimpleNamespace();namespace['options']([output])
            flags=next(iter(output.compile_options))
            tokens=shlex.split(flags.removeprefix('FLAGS:'))
            # Actual include and link search paths remain intact after shell
            # tokenization, but no directory can be mistaken for a source file.
            self.assertIn('-I'+str(root/'native'),tokens)
            self.assertIn('-L'+str(root/'native/build'),tokens)
            self.assertIn('-rpath='+str(root/'native/build'),tokens)
            self.assertTrue(all(t.startswith('-') for t in tokens),tokens)


if __name__=='__main__':unittest.main(verbosity=2)

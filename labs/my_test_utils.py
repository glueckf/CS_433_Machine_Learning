"""Drop-in replacement for the labs' ``test_utils.test`` with float tolerance.

The course doctests compare printed output character by character, which breaks
in two harmless ways on a modern setup:

* NumPy 2 prints scalars as ``np.float64(0.25)`` instead of ``0.25``.
* Results differ in the last few digits depending on the BLAS/LAPACK backend
  (e.g. Apple Accelerate), so ``0.2895728056656946`` may come out as
  ``0.2895728056657044``.

This checker strips the NumPy scalar wrappers and compares every number with a
relative/absolute tolerance; all non-numeric text must still match exactly.

Usage in a notebook (from ``labs/exXX/template``)::

    import sys
    sys.path.append("../..")
    from my_test_utils import test
"""

import doctest
import io
import re
import sys

import numpy as np

RTOL = 1e-9
ATOL = 1e-12

_NP_SCALAR = re.compile(r"np\.(?:float|int|uint|bool_?)\d*\((.*?)\)")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _normalize(text):
    """Remove NumPy 2 scalar wrappers and collapse whitespace."""
    text = _NP_SCALAR.sub(r"\1", text)
    return " ".join(text.split())


def _close_enough(want, got):
    """Same text skeleton, and every number pairwise close."""
    want, got = _normalize(want), _normalize(got)
    if _NUMBER.sub("#", want) != _NUMBER.sub("#", got):
        return False
    want_nums = [float(n) for n in _NUMBER.findall(want)]
    got_nums = [float(n) for n in _NUMBER.findall(got)]
    return np.allclose(want_nums, got_nums, rtol=RTOL, atol=ATOL)


class _TolerantChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        if super().check_output(want, got, optionflags):
            return True
        return _close_enough(want, got)


def test(f):
    """Run the doctests in ``f``'s docstring, tolerating float noise."""
    tests = doctest.DocTestFinder().find(f)
    assert len(tests) <= 1
    for t in tests:
        orig_stdout = sys.stdout
        sys.stdout = io.StringIO()
        orig_rng_state = np.random.get_state()
        try:
            np.random.seed(1)
            runner = doctest.DocTestRunner(checker=_TolerantChecker())
            results = runner.run(t)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = orig_stdout
            np.random.set_state(orig_rng_state)

        if results.failed > 0:
            print(f"❌ The are some issues with your implementation of `{f.__name__}`:")
            print(output, end="")
            print("*" * 70)
        elif results.attempted > 0:
            print(f"✅ Your `{f.__name__}` passes some basic tests.")
        else:
            print(f"Could not find any tests for {f.__name__}")

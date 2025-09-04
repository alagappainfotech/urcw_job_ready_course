"""
Module 8: Pythonic Best Practices & Continuous Improvement - Exercises and Labs
"""

import re


# Quick Checks

def qc_pep8_name_checks():
    names = [
        "UserName", "user_name", "MAX_SIZE", "getUserData", "process_data",
    ]
    print("Evaluate names (PEP 8):", names)


# Try This

def try_docstrings_and_types():
    def add(a: int, b: int) -> int:
        """Return the sum of two integers."""
        return a + b

    print("add(2, 3) ->", add(2, 3))


# Labs

def lab_refactor_example(code: str):
    """Very basic refactor: collapse multiple spaces, trim lines."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in code.splitlines()]
    return "\n".join(lines)


def try_profiling_examples():
    import timeit
    setup = "nums = list(range(10000))"
    s1 = "sum(nums)"
    t = timeit.timeit(stmt=s1, setup=setup, number=100)
    print("timeit sum(nums) x100:", t)
    try:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        sum(range(1000000))
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats(5)
        print("cProfile top5:\n", s.getvalue())
    except Exception as e:
        print("Profiling not available:", e)


if __name__ == "__main__":
    qc_pep8_name_checks()
    try_docstrings_and_types()
    sample = """
    def   add ( a ,  b ):
        return   a +  b
    """
    print("Refactor preview:\n", lab_refactor_example(sample))
    try_profiling_examples()



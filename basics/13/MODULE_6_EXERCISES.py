"""
Module 6: Packages & Ecosystem - Exercises and Labs
"""

def qc_imports_and_paths():
    import sys, pkgutil
    print("Top 5 sys.path entries:")
    for p in sys.path[:5]:
        print(" ", p)
    names = sorted([m.name for m in pkgutil.iter_modules()])[:20]
    print("Installed (first 20):", names)


def try_package_layout_example(tmp_dir: str = "./demo_pkg"):
    from pathlib import Path
    root = Path(tmp_dir)
    (root / "processing").mkdir(parents=True, exist_ok=True)
    (root / "io").mkdir(parents=True, exist_ok=True)

    files = {
        root / "__init__.py": "__all__ = ['utils']\n_private_var = 42\n",
        root / "utils.py": "def normalize_name(s): return s.strip().lower()\n",
        root / "processing" / "__init__.py": "",
        root / "processing" / "cleaning.py": "def clean(s): return ' '.join(s.split())\n",
        root / "io" / "__init__.py": "",
        root / "io" / "readers.py": "def read_lines(p): return open(p).read().splitlines()\n",
    }
    for path, content in files.items():
        path.write_text(content)
    print(f"Package skeleton created at {root.resolve()}")

    # Demonstrate private name pattern
    import sys
    sys.path.insert(0, str(Path('.').resolve()))
    modname = root.name
    pkg = __import__(modname)
    print("Public API:", getattr(pkg, "__all__", []))


def lab_dependency_report(requirements_path: str = "requirements.txt"):
    from pathlib import Path
    if not Path(requirements_path).exists():
        print("No requirements.txt found.")
        return
    print("Dependencies in requirements.txt:")
    for line in Path(requirements_path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            print(" ", s)


if __name__ == "__main__":
    qc_imports_and_paths()
    try_package_layout_example()
    lab_dependency_report()



import importlib

required_modules = {
    "tensorflow": "tensorflow",
    "numpy": "numpy",
    "pillow": "PIL",              # Correct module name for Pillow
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit_learn": "sklearn",
    "tkinter": "tkinter"          # Only works on systems where Tkinter is built-in
}

missing = []

for name, module in required_modules.items():
    try:
        importlib.import_module(module)
    except ImportError:
        missing.append(name)

if not missing:
    print("✅ All required libraries are installed.")
else:
    print("❌ Missing libraries:")
    for lib in missing:
        print(f" - {lib}")

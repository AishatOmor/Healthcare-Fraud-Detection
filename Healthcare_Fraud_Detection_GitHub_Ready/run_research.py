from pathlib import Path
import subprocess, sys
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parent
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    print("Running", path.name, flush=True)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}}).execute()
for path in sorted((ROOT / "src").glob("*.py")):
    subprocess.run([sys.executable,str(path),"--data",str(ROOT/"data"),"--out",str(ROOT/"outputs")],check=True,cwd=ROOT)

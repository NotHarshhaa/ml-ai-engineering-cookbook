import ast
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class NotebookIntegrityTests(unittest.TestCase):
    def test_all_notebooks_are_valid_json_and_nbformat_four(self):
        notebooks = sorted(REPOSITORY_ROOT.rglob("*.ipynb"))
        self.assertGreaterEqual(len(notebooks), 60, "Expected the cookbook notebooks")

        for path in notebooks:
            with self.subTest(notebook=path.relative_to(REPOSITORY_ROOT)):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook.get("nbformat"), 4)
                self.assertIsInstance(notebook.get("cells"), list)

    def test_all_notebook_code_cells_parse_as_python(self):
        for path in sorted(REPOSITORY_ROOT.rglob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell_number, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                with self.subTest(
                    notebook=path.relative_to(REPOSITORY_ROOT), cell=cell_number
                ):
                    ast.parse(source, filename=f"{path} (cell {cell_number})")


if __name__ == "__main__":
    unittest.main()

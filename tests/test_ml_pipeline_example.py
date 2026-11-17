import ast
import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "12-mlops" / "03_ml_pipelines.ipynb"


def load_mini_pipeline_class():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        tree = ast.parse(source)
        class_node = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name == "MiniPipeline"),
            None,
        )
        if class_node is not None:
            namespace = {
                "Path": Path,
                "hashlib": hashlib,
                "pickle": pickle,
            }
            module = ast.Module(body=[class_node], type_ignores=[])
            exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
            return namespace["MiniPipeline"]
    raise AssertionError("MiniPipeline class was not found in its notebook")


MiniPipeline = load_mini_pipeline_class()


class MiniPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.pipeline = MiniPipeline(cache_dir=self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_run_returns_only_the_requested_output(self):
        self.pipeline.add_node(
            "source", lambda value: value * 2, inputs=["input"],
            outputs=["result", "unused"],
        )

        self.assertEqual(self.pipeline.run("result", input=5), {"result": 10})

    def test_missing_target_raises_a_clear_key_error(self):
        self.pipeline.add_node("source", lambda: 1, inputs=[], outputs=["result"])

        with self.assertRaisesRegex(KeyError, "missing"):
            self.pipeline.run("missing")

    def test_cache_key_changes_when_input_values_change(self):
        self.pipeline.add_node(
            "double", lambda value: value * 2, inputs=["value"], outputs=["result"]
        )

        first = self.pipeline.run("result", value=3)
        second = self.pipeline.run("result", value=4)

        self.assertEqual(first, {"result": 6})
        self.assertEqual(second, {"result": 8})


if __name__ == "__main__":
    unittest.main()

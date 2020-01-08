import os
import shutil

from marmot.adapter.adapter import DataAdapter


class TestAdapter:

    def setup_method(self):
        self.adapter = DataAdapter()
        self.tempdir = os.path.join("~", "temp")
        self.path_to_save = os.path.join(self.tempdir, "adapter_config.json")
        os.makedirs(self.tempdir)

    def teardown_method(self):
        del self.adapter
        shutil.rmtree(self.tempdir)

    def test_save_config(self):
        self.adapter.save(self.path_to_save)

    def test_load_confg(self):
        self.adapter.config.explainers = ["A", "B", "C"]
        self.adapter.config.objectives = ["D", "E"]
        self.adapter.save(self.path_to_save)
        adapter = DataAdapter.from_json(self.path_to_save)

        assert adapter.config.explainers == ["A", "B", "C"]
        assert adapter.config.objectives == ["D", "E"]

    def test_reload_config(self):
        self.adapter.config.explainers = ["A", "B", "C"]
        self.adapter.config.objectives = ["D", "E"]
        self.adapter.save(self.path_to_save)
        adapter = DataAdapter()
        adapter.reload_config(self.path_to_save)
        assert adapter.config.explainers == ["A", "B", "C"]
        assert adapter.config.objectives == ["D", "E"]


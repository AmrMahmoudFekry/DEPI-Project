import os

class StorageManager:
    def __init__(self, base_path="."): 
        self.base_path = base_path

    def _get_full_path(self, filename):
        return os.path.join(self.base_path, filename)

    def read_all(self, filename):
        path = self._get_full_path(filename)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as file:
            return file.readlines()

    def append(self, filename, content):
        path = self._get_full_path(filename)
        with open(path, "a", encoding="utf-8") as file:
            file.write(str(content) + "\n")
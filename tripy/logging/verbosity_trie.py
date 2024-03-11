import os
from typing import Optional, Set, Dict


class VerbosityTrie(dict):
    """
    A trie that contains per-path logging verbosities.
    Paths should be relative to the tripy module root directory.
    """

    def _split_path(self, path):
        # Ignore empty elements created by leading or duplicate slashes in the path components.
        return list(filter(lambda x: x, path.split(os.path.sep)))

    def __init__(self, verbosity_map: Dict[str, Set[str]]):
        assert "" in verbosity_map, "verbosity_map must include default verbosity!"

        for path, verbosity_set in verbosity_map.items():
            cur_map = self
            for path_component in self._split_path(path):
                if path_component not in cur_map:
                    cur_map[path_component] = {}
                cur_map = cur_map[path_component]

            cur_map[""] = verbosity_set  # Add default verbosity set

        assert "" in self  # Should have added a default key.
        # Skip path checking if we don't have any path entries.
        self.has_non_default_entries = len(self) > 1

    def get_verbosity_set(self, path: Optional[str] = None) -> Set[str]:
        """
        Get the logging verbosity set for the given path.

        Args:
            path: The path. If this is None, the default verbosity set is returned.

        Returns:
            The logging verbosity set.
        """
        default_verbosity_set = self[""]
        if path is None or not self.has_non_default_entries:
            return default_verbosity_set

        cur_map = self

        def get_value(dct):
            return dct.get("", default_verbosity_set)

        for path_component in self._split_path(path):
            if path_component not in cur_map:
                return get_value(cur_map)
            cur_map = cur_map[path_component]
        return get_value(cur_map)

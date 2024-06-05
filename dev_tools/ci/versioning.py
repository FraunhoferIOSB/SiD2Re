# import typing
# from pathlib import Path
#
# import tomli
import fire
# import re
# import semver
# from semver import VersionInfo


# def parse_version(version: str) -> VersionInfo:
#     return VersionInfo.parse(version)


SCOPE_MAP = dict(
    feature="minor",
    patch="patch",
    minor="minor",
    major="major",
    docs="patch",
    test="patch",
    ci="patch",
    fix="patch",
)

class VersionUpdater:

    # def _read_toml(self, pyproject_toml: Path):
    #     with open(pyproject_toml, "r") as f:
    #         return f.read()

    # def _get_toml(self, pyproject_toml: str) -> typing.Dict:
    #     return tomli.loads(pyproject_toml)

    # def _write_toml(self, path: Path, pyproject_toml: str):
    #     with open(path, "w") as f:
    #         f.write(pyproject_toml)

    # def set_version(self, path: Path, version: str):
    #     pyproject_toml = self._read_toml(path)
    #     pyproject_toml = self._set_version(pyproject_toml, version)
    #     self._write_toml(path, pyproject_toml)

    # def _set_version(self, pyproject_toml: str, version: VersionInfo) -> str:
    #     current_version = str(self._get_version(pyproject_toml))
    #     return re.sub(rf"^version = \"{current_version}\"", f"version = \"{version}\"", pyproject_toml,
    #                   flags=re.MULTILINE)
    #     # pyproject_toml = pyproject_toml.replace(f"version = \"{current_version}\"", f"version = \"{version}\"")
    #     # return pyproject_toml

    # def _get_version(self, pyproject_toml: str) -> VersionInfo:
    #     toml = self._get_toml(pyproject_toml)
    #     return parse_version(toml["tool"]["poetry"]["version"])

    # def get_version(self, path: Path) -> str:
    #     pyproject_toml = self._read_toml(path)
    #     return str(self._get_version(pyproject_toml))

    # def bump_minor(self, path: Path):
    #     self.bump("minor", path)

    # def bump_feature(self, path: Path):
    #     self.bump_minor(path)

    # def bump_patch(self, path: Path):
    #     self.bump("minor", path)

    # def bump_major(self, path: Path):
    #     self.bump("major", path)

    # def bump(self, part: str, path: Path):
    #     pyproject_toml = self._read_toml(path)
    #     version = self._get_version(pyproject_toml)
    #     new_version = version.next_version(part=part)
    #     updated_pyproject_toml = self._set_version(pyproject_toml, new_version)
    #     self._write_toml(path, updated_pyproject_toml)
    #     return f"changed from version {version} -> {new_version}"

    def retrieve_part(self, branch_name: str) -> str:
        scope = branch_name.split("/")[0]
        return SCOPE_MAP.get(scope, "patch")


if __name__ == "__main__":

    fire.Fire(VersionUpdater())

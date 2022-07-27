import difflib
import json
import os
import pprint
import subprocess
import sys
import time
from collections import OrderedDict
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
)

import git
import javalang

os.chdir("/Users/bohrok/Documents/replication-kit-2020-line-validation")
pp = pprint.PrettyPrinter(indent=4)


REPODICT = {
    "ivy": {
        "repo_path": "data/repos/ant-ivy",
        "data_dir": "data/parsed/ant-ivy",
    },
    "math": {
        "repo_path": "data/repos/commons-math",
        "data_dir": "data/parsed/commons-math",
    },
    "opennlp": {
        "repo_path": "data/repos/opennlp",
        "data_dir": "data/parsed/opennlp",
    }
}


def get_parents(commit: git.Commit) -> Sequence[git.Commit]:
    return commit.parents


def get_diff(new: git.Commit, old: git.Commit) -> git.DiffIndex:
    return new.diff(old)


def get_blobs(diff: git.Diff) -> tuple:
    return diff.a_blob, diff.b_blob


def get_changed_lines(
    a_blob: git.Blob, b_blob: git.Blob, repo_path: str
) -> Tuple[List[int], List[int]]:
    """
    return (line_num_a, line_num_b)
    line_num_a: a blob에서 삭제된 line_num 들
    line_num_b: b blob에서 추가된 line_num 들
    # (취소) line_num_m: a 에서 b로 바뀐 line_num 들 (pair)
    현재 사용 기준, a_blob은 old blob, b_blob은 new blob
    """
    a_blob_str = subprocess.check_output(
        f"git cat-file -p {a_blob.hexsha}", shell=True, cwd=repo_path
    ).decode("utf-8", "backslashreplace")
    b_blob_str = subprocess.check_output(
        f"git cat-file -p {b_blob.hexsha}", shell=True, cwd=repo_path
    ).decode("utf-8", "backslashreplace")
    s = difflib.SequenceMatcher(
        None, a_blob_str.splitlines(), b_blob_str.splitlines()
    )
    line_num_a, line_num_b = [], []
    for tag, a1, a2, b1, b2 in s.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "delete":  # a에선 없고 b에선 있는 line
            line_num_a.extend(range(a1 + 1, a2 + 1))
        elif tag == "insert":  # a에선 있고, b에선 없는 line
            line_num_b.extend(range(b1 + 1, b2 + 1))
        elif tag == "replace":  # a에서 b로 바뀌는 line
            line_num_a.extend(range(a1 + 1, a2 + 1))
            line_num_b.extend(range(b1 + 1, b2 + 1))
            # assert a2 - a1 == b2 - b1
            # line_num_m.extend(zip(range(a1 + 1, a2 + 1), range(b1 + 1, b2 + 1)))
    return line_num_a, line_num_b


def get_changes(
    diff: git.Diff, repo_path: str
) -> Tuple[str, List[int], List[int]]:
    """
    return: (file, line_num_new, line_num_old)
    line_num_new: commit 이후 버전 기준 추가된 line_num 들
    line_num_old: commit 이전 버전 기준 삭제된 line_num 들
    # (취소) line_num_mod: commit 이전 버전에서 이후 버전으로 바뀐 line_num 들 (pair)
    """
    new_blob, old_blob = get_blobs(diff)
    file_path = new_blob.path
    line_num_old, line_num_new = get_changed_lines(
        old_blob, new_blob, repo_path
    )
    return file_path, line_num_old, line_num_new


def is_modified_file(diff: git.Diff) -> bool:
    return (
        not diff.deleted_file
        and not diff.new_file
        and not diff.renamed_file
        and not diff.copied_file
    )


class DiffExplodeException(Exception):
    pass


class BlobNotFoundException(Exception):
    pass


class Changes:
    def __init__(self, repo_path: str, commit: git.Commit):
        self.repo_path = repo_path
        self.commit = commit
        self.author = commit.author.name
        self.cid = commit.hexsha
        self.parents = {}
        for parent in get_parents(commit):
            pcid = parent.hexsha
            self.parents[pcid] = parent
        self.diffs = {}
        for pcid, parent in self.parents.items():
            self.diffs[pcid] = {}
            print(f"{len(list(get_diff(commit, parent)))=}")
            if len(list(get_diff(commit, parent))) > 1000:
                raise DiffExplodeException(
                    f"{pcid} has too many diffs ({len(list(get_diff(commit, parent)))})"
                )
            for diff in get_diff(commit, parent):
                if not is_modified_file(diff):
                    continue
                (
                    file_path,
                    line_num_old,
                    line_num_new,
                    # line_num_mod,
                ) = get_changes(diff, repo_path)
                self.diffs[pcid][file_path] = (
                    line_num_old,
                    line_num_new,
                    # line_num_mod,
                )

    def get_blob(self, pcid, file_path, is_old):
        parent = self.parents[pcid]
        for diff in get_diff(commit, parent):
            if (diff.a_blob and diff.a_blob.path == file_path) or (
                diff.b_blob and diff.b_blob.path == file_path
            ):
                return diff.b_blob if is_old else diff.a_blob
        raise BlobNotFoundException(
            f"{file_path}:{is_old=} not found in {pcid}"
        )

    def print_context_diff(self, pcid, file_path):
        old_blob = self.get_blob(pcid, file_path, True)
        new_blob = self.get_blob(pcid, file_path, False)
        old_blob_str = subprocess.check_output(
            f"git cat-file -p {old_blob.hexsha}", shell=True, cwd=self.repo_path
        ).decode("utf-8", "backslashreplace")
        new_blob_str = subprocess.check_output(
            f"git cat-file -p {new_blob.hexsha}", shell=True, cwd=self.repo_path
        ).decode("utf-8", "backslashreplace")
        diff_result = difflib.context_diff(
            [s + "\n" for s in old_blob_str.splitlines()],
            [s + "\n" for s in new_blob_str.splitlines()],
            n=3,
        )
        sys.stdout.writelines(diff_result)


import contextlib


def is_java_file(file_path: str) -> bool:
    return file_path.endswith(".java")


def is_merge_commit(commit: git.Commit) -> bool:
    return len(commit.parents) > 1


def parse_java_file(file_str: str) -> javalang.tree.CompilationUnit:
    return javalang.parse.parse(file_str)


def get_final_line(node: javalang.tree.Node) -> int:
    # traverse node and find the max line
    max_line = 0
    for path, child in node.filter(javalang.tree.Node):
        with contextlib.suppress(TypeError):
            max_line = max(max_line, child.position[0])
    return max_line


def build_position_dict(tree: javalang.tree.CompilationUnit) -> Dict:
    package_name = None if tree.package is None else tree.package.name
    ret = {"package": package_name, "classes": []}
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        classDeclNode: javalang.tree.ClassDeclaration = node
        class_name = classDeclNode.name
        class_pos = (
            classDeclNode.position[0],
            get_final_line(classDeclNode),
        )
        class_dict = {"name": class_name, "pos": class_pos, "methods": []}
        for method in classDeclNode.methods:
            method_dict = {
                "name": method.name,
                "pos": (method.position[0], get_final_line(method)),
                "paramtypes": [p.type.name for p in method.parameters],
            }
            class_dict["methods"].append(method_dict)
        class_dict["inner"] = None
        for prev_class_dict in ret["classes"]:
            if (
                prev_class_dict["pos"][1] >= class_pos[1]
                and prev_class_dict["pos"][0] <= class_pos[0]
            ):
                class_dict["inner"] = prev_class_dict["name"]
                break
        ret["classes"].append(class_dict)
    return ret


def get_posdict_from_blob(blob: git.Blob) -> Dict:
    file_str = subprocess.check_output(
        f"git cat-file -p {blob.hexsha}", shell=True, cwd=repo_path
    ).decode("utf-8", "backslashreplace")
    tree: javalang.tree.CompilationUnit = parse_java_file(file_str)
    return build_position_dict(tree)


def get_clsNmeth(pos_dict: Dict, line_num: int) -> Tuple[str, Optional[str]]:
    package = pos_dict["package"]
    clspath, methsig = None, None
    for cls in pos_dict["classes"]:
        if cls["pos"][0] <= line_num <= cls["pos"][1]:
            clsname = cls["name"]
            inner = cls["inner"]
            while inner:
                clsname = f"{inner}#{clsname}"
                outer_class = [
                    c for c in pos_dict["classes"] if c["name"] == inner
                ][0]
                inner = outer_class["inner"]
            clspath = f"{package}#{clsname}"
            for method in cls["methods"]:
                if method["pos"][0] <= line_num <= method["pos"][1]:
                    methsig = method["name"]
                    if method["paramtypes"]:
                        methsig = f"{methsig}({','.join(method['paramtypes'])})"
                    break
            break
    return (clspath, methsig)


def get_change_dict(
    changed_line_nums: List[int],
    changes: Changes,
    pcid: str,
    file_path: str,
    is_old: bool,
) -> OrderedDict:
    change_dict = OrderedDict()
    blob = changes.get_blob(pcid, file_path, is_old)
    pos_dict = get_posdict_from_blob(blob)
    for line_num in changed_line_nums:
        change_clspath, change_methname = get_clsNmeth(pos_dict, line_num)
        if str((change_clspath, change_methname)) not in change_dict:
            change_dict[str((change_clspath, change_methname))] = []
        change_dict[str((change_clspath, change_methname))].append(line_num)
    return change_dict


if __name__ == "__main__":
    assert len(sys.argv) == 2
    repo_name = sys.argv[1]
    repo_path = REPODICT[repo_name]["repo_path"]
    data_dir = REPODICT[repo_name]["data_dir"]
    repo = git.Repo(repo_path)
    parsed_data = OrderedDict()
    size = len(list(repo.iter_commits()))
    print(f"{size} commits")
    for idx, commit in enumerate(repo.iter_commits(), 1):
        group_idx = ((idx - 1) // 10 + 1) * 10
        if os.path.exists(os.path.join(data_dir, f"{str(group_idx)}.json")):
            print(
                "Skip group", group_idx, f"because it already exists ({idx=})"
            )
            continue
        # flush currently parsed data
        if (idx - 1) % 10 == 0 and idx != 1:
            print(f"{idx}/{size}")
            with open(os.path.join(data_dir, f"{idx - 1}.json"), "w") as f:
                json.dump(parsed_data, f, indent=4)
                parsed_data = OrderedDict()
        parsed_commit_data = OrderedDict()
        parsed_commit_data["authored_data"] = time.strftime(
            "%Y %b %d %H:%M", time.gmtime(commit.authored_date)
        )
        parsed_commit_data["commit.message"] = commit.message
        parsed_commit_data["commit.author.name"] = commit.author.name
        if is_merge_commit(commit):
            continue
        try:
            changes = Changes(repo_path, commit)
        except DiffExplodeException as e:
            print(e)
            continue
        assert len(changes.diffs) < 2
        if not changes.diffs:
            continue
        pcid, diff_dict = changes.diffs.popitem()
        parsed_commit_data["pcid"] = pcid
        change_dict = OrderedDict()
        for file_path, (line_num_old, line_num_new) in diff_dict.items():
            file_change_dict = OrderedDict()
            if not is_java_file(file_path):
                continue
            if len(line_num_old):
                try:
                    file_change_dict["old"] = get_change_dict(
                        line_num_old, changes, pcid, file_path, True
                    )
                except (
                    javalang.parser.JavaSyntaxError,
                    javalang.tokenizer.LexerError,
                ) as e:
                    file_change_dict["old"] = (
                        e.__class__.__name__,
                        file_path,
                        line_num_old,
                    )
            if len(line_num_new):
                try:
                    file_change_dict["new"] = get_change_dict(
                        line_num_new, changes, pcid, file_path, False
                    )
                except (
                    javalang.parser.JavaSyntaxError,
                    javalang.tokenizer.LexerError,
                ) as e:
                    file_change_dict["new"] = (
                        e.__class__.__name__,
                        file_path,
                        line_num_new,
                    )
            if len(file_change_dict):
                change_dict[file_path] = file_change_dict
        if not len(change_dict):
            continue
        parsed_commit_data["changes"] = change_dict
        parsed_data[commit.hexsha] = parsed_commit_data
    if len(parsed_data):
        with open(os.path.join(data_dir, f"{idx}.json"), "w") as f:
            json.dump(parsed_data, f, indent=4)

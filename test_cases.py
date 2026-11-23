import operator
import re
import unittest
from collections import defaultdict

from cases import make_cases
from route import features, predict


def reference(question, family):
    if family == "arithmetic":
        start, body = re.search(r"Start at (\d+)\..*precedence: (.*?)\.", question).groups()
        value = int(start)
        ops = {"+": operator.add, "-": operator.sub, "*": operator.mul}
        for op, number in re.findall(r"([+*\-]) (\d+)", body):
            value = ops[op](value, int(number))
        return value
    if family == "program":
        a, b, code, target = re.search(
            r"a=(\d+) and b=(\d+).*in order: (.*?)\. What is the final value of ([ab])", question
        ).groups()
        assert all(line in {"a,b=b,a", "a=a+b", "b=b-a", "b=a+2*b", "a=a-2*b"} for line in code.split("; "))
        state = {"a": int(a), "b": int(b)}
        exec(code, {"__builtins__": {}}, state)
        return state[target]
    if family == "graph":
        edges = re.findall(r"([A-Z])->([A-Z])", question)
        start, end = re.search(r"from ([A-Z]) to ([A-Z])", question).groups()
        reachable = {start}
        for distance in range(1, len(edges) + 1):
            reachable |= {v for u, v in edges if u in reachable}
            if end in reachable:
                return distance
        raise AssertionError("生成题无解")
    rows = re.findall(r"\(([A-Z]),([A-Z]),(\d+)\)", question)
    group, kind = re.search(r"group=([A-Z]) and kind=([A-Z])", question).groups()
    return sum(int(n) for g, k, n in rows if g == group and k == kind)


class CasesTest(unittest.TestCase):
    def test_all_gold_answers_against_reference(self):
        ids = set()
        answers_by_size = defaultdict(set)
        for row in make_cases():
            self.assertNotIn(row["id"], ids)
            ids.add(row["id"])
            choices = {k: int(v) for k, v in re.findall(r"^([A-D])\. (-?\d+)$", row["prompt"], re.M)}
            self.assertEqual(len(set(choices.values())), 4)
            self.assertEqual(reference(row["prompt"], row["family"]), choices[row["answer"]])
            answers_by_size[(row["split"], row["family"], row["size"])].add(row["answer"])
        self.assertEqual(len(ids), 608)
        self.assertTrue(all(len(options) > 1 for options in answers_by_size.values()))

    def test_no_answer_features_and_safe_fallback(self):
        p = "Solve.\n\nStart at 3.\nA. 1\nB. 3"
        self.assertEqual(features(p), features(p.replace("B. 3", "B. 5000")))
        model = {
            "features": [], "heads": [{"intercept": 0, "coefficients": []}] * 5,
            "depths": [1, 2, 3, 4, 5], "policy": "fixed-3",
        }
        self.assertEqual(predict(model, p, 10)["depth"], 3)


if __name__ == "__main__":
    unittest.main()

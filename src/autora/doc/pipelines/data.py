from typing import Iterable, List, Tuple


def load_data(data_file: str) -> Tuple[List[str], List[str]]:
    import jsonlines

    with jsonlines.open(data_file) as reader:
        items = [item for item in reader]
        inputs = [item["instruction"] for item in items]
        labels = [item["output"] for item in items]
        return inputs, labels


def preprocess_code(code: str) -> str:
    lines: Iterable[str] = code.splitlines()
    skip_starts = {"import", "from", "#"}
    lines = filter(
        lambda line: not (any([line.strip().startswith(skip) for skip in skip_starts]) or line.strip() == ""),
        lines,
    )
    return "\n".join(lines)

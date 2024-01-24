from __future__ import annotations

from enum import Enum

LLAMA2_INST_CLOSE = "[/INST]\n"
CODE_PLACEHOLDER = "__CODE_INPUT__"


class PromptBuilder:
    """
    Utilty class for building LLAMA2 prompts. Uses a stateful builder pattern.
    See: https://ai.meta.com/llama/get-started/#prompting
    """

    def __init__(self, sys: str, instr: str):
        self.instr = instr
        # Initialize the prompt with the system prompt
        self.prompt_text = PromptBuilder._trim_leading_ws(
            f"""
            <s>[INST] <<SYS>>
            { sys }
            <</SYS>>
            """
        )

    def _add_input(self) -> PromptBuilder:
        # Add the instruction (e.g. "Generate a one line descrip...")
        # and a placeholder for the code
        self.prompt_text += PromptBuilder._trim_leading_ws(
            f"""
            { self.instr }
            ----------
            {CODE_PLACEHOLDER}
            ----------
            """
        )
        return self

    def add_example(self, code: str, doc: str) -> PromptBuilder:
        # This adds an example in the form of instruction+code+doc
        self._add_input()
        self.prompt_text = self.prompt_text.replace(CODE_PLACEHOLDER, code)
        self.prompt_text += PromptBuilder._trim_leading_ws(
            f"""
            [/INST]
            {doc}
            </s>
            <s>
            [INST]
            """
        )
        return self

    def build(self) -> str:
        # Add a instruction+code placeholder and close the instruction
        self._add_input()
        self.prompt_text = self.prompt_text + LLAMA2_INST_CLOSE
        return self.prompt_text

    @staticmethod
    def _trim_leading_ws(s: str) -> str:
        return "\n".join([line.lstrip() for line in s.splitlines()])


SYS_1 = """You are a technical documentation writer. You always write clear, concise, and accurate documentation for
scientific experiments. Your documentation focuses on the experiment's purpose, procedure, and results. Therefore,
details about specific python functions, packages, or libraries are not necessary. Your readers are experimental
scientists.
"""

SYS_GUIDES = """You are a technical documentation writer. You always write clear, concise, and accurate documentation
for scientific experiments. Your documentation focuses on the experiment's procedure. Therefore, details about specific
python functions, packages, or libraries are NOT necessary. Your readers are experimental scientists.
For writing your descriptions, follow these instructions:
- DO NOT write greetings or preambles
- Use the Variable 'name' attribute and not the python variable names
- Use LaTeX for math expressions
- DO NOT include code or code-like syntax and do not use python function or class names
- Write in paragraph style, NOT bullet points
"""

INSTR_SWEETP_1 = (
    """Please generate high-level one or two paragraph documentation for the following experiment."""
)


INSTR_AUTORA_VARS = """Generate a one line description of the dependent and independent variables used in the following
python code: """

CODE_AUTORA_VARS1 = """
iv1 = Variable(name="a", value_range=(0, 2 * np.pi), allowed_values=np.linspace(0, 2 * np.pi, 30))
iv2 = Variable(name="b", value_range=(0, 1), allowed_values=np.linspace(0, 1, 30))
dv = Variable(name="z", type=ValueType.REAL)
variables = VariableCollection(independent_variables=[iv1, iv2], dependent_variables=[dv])
"""

DOC_AUTORA_VARS1 = """The problem is defined by two independent variables $a \in [0, 2 \pi]$, $b in [0,1] and a
dependent variable $z$."""


class PromptIds(str, Enum):
    SWEETP_1 = "SWEETP_1"
    AUTORA_VARS_ZEROSHOT = "AUTORA_VARS_ZEROSHOT"
    AUTORA_VARS_ONESHOT = "AUTORA_VARS_ONESHOT"


PROMPTS = {
    PromptIds.SWEETP_1: PromptBuilder(SYS_1, INSTR_SWEETP_1).build(),
    PromptIds.AUTORA_VARS_ZEROSHOT: PromptBuilder(SYS_GUIDES, INSTR_AUTORA_VARS).build(),
    PromptIds.AUTORA_VARS_ONESHOT: PromptBuilder(SYS_GUIDES, INSTR_AUTORA_VARS)
    .add_example(CODE_AUTORA_VARS1, DOC_AUTORA_VARS1)
    .build(),
}

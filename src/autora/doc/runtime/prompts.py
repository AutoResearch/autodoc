from enum import Enum

LLAMA2_INST_CLOSE = "[/INST]\n"

# Standard Llama2 template
TEMP_LLAMA2 = """
[INST]<<SYS>>
{sys}

{instr}

{input}
[/INST]
"""


SYS_1 = """You are a technical documentation writer. You always write clear, concise, and accurate documentation for
scientific experiments. Your documentation focuses on the experiment's purpose, procedure, and results. Therefore,
details about specific python functions, packages, or libraries are not necessary. Your readers are experimental
scientists.
"""

INSTR_SWEETP_1 = """Please generate high-level two paragraph documentation for the following experiment. The first
paragraph should explain the purpose and the second one the procedure, but don't use the word 'Paragraph'"""

# The following prompt uses an example (code, doc) to specify the desired behavior
EX_CODE = """
from sweetpea import *

color = Factor('color', ['red', 'green', 'blue', 'yellow'])
word = Factor('word', ['red', 'green', 'blue', 'yellow'])

def is_congruent(word, color):
    return (word == color)

def is_not_congruent(word, color):
    return not is_congruent(word, color)

congruent = DerivedLevel('congruent', WithinTrial(is_congruent, [word, color]))
incongruent = DerivedLevel('incongruent', WithinTrial(is_not_congruent, [word, color]))

congruency = Factor('congruency', [congruent, incongruent])

constraints = [MinimumTrials(48)]
design = [word, color, congruency]
crossing = [word, congruency]

block = CrossBlock(design, crossing, constraints)

experiment = synthesize_trials(block, 1)

save_experiments_csv(block, experiment, 'code_1_sequences/seq')
"""

EX_DOC = """There are two regular factors: color and word. The color factor consists of four levels: "red", "green",
"blue", and "yellow". The word factor also consists of the four levels: "red", "green", "blue", and "yellow".
There is another derived factor referred to as congruency. The congruency factor depends on the regular factors word
and color and has two levels: "congruent" and "incongruent". A trial is considered "congruent" if the word matches
the color, otherwise, it is considered "incongruent". We counterbalanced the word factor with the congruency factor.
All experiment sequences contained at least 48 trials."""

INSTR_SWEETP_EXAMPLE = f"""Consider the following experiment code:
---
{EX_CODE}
---
Here's a a good English description:
---
{EX_DOC}
---
Using the same style, please generate a high-level one paragraph description for the following experiment code:
"""


AUTORA_VARS = (
    "Generate a high level description of the variables used in the experiment based on the following code. "
    "Do not include code or code-like syntax in your description."
)


class SystemPrompts(str, Enum):
    SYS_1 = "SYS_1"


class InstructionPrompts(str, Enum):
    INSTR_SWEETP_1 = "INSTR_SWEETP_1"
    INSTR_SWEETP_EXAMPLE = "INSTR_SWEETP_EXAMPLE"


SYS = {SystemPrompts.SYS_1: SYS_1}
INSTR = {
    InstructionPrompts.INSTR_SWEETP_1: INSTR_SWEETP_1,
    InstructionPrompts.INSTR_SWEETP_EXAMPLE: INSTR_SWEETP_EXAMPLE,
}

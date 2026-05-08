"""
3806ICT Assignment 1: First-Order Logic Parser and Sequent Calculus Proof Search

This module implements:
1. A parser for first-order logic formulae in course notation or TPTP format
2. An AST representation for first-order logic formulae
3. Substitution and unification for terms and formulas
4. Algorithm 2: A tableau/sequent calculus based proof search method

Author: s5305835
Date: 2026
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union


class ParserError(ValueError):
    """Exception raised when parsing fails."""
    pass


# ============================================================================
# AST Classes - Representing First-Order Logic Formulae
# ============================================================================

class Term:
    """Represents a term (variable or constant function application)."""
    
    def __init__(self, name: str, args: Tuple["Term", ...] = ()) -> None:
        self.name = name
        self.args = args


class Predicate:
    """Represents a predicate (atomic formula)."""
    
    def __init__(self, name: str, args: Tuple[Term, ...] = ()) -> None:
        self.name = name
        self.args = args


class Truth:
    """Represents the constant ⊤ (true)."""
    
    def __repr__(self) -> str:
        return "Truth()"


class Falsehood:
    """Represents the constant ⊥ (false)."""
    
    def __repr__(self) -> str:
        return "Falsehood()"


class Not:
    """Represents negation: ¬φ."""
    
    def __init__(self, operand: "Formula") -> None:
        self.operand = operand


class And:
    """Represents conjunction: φ ∧ ψ."""
    
    def __init__(self, left: "Formula", right: "Formula") -> None:
        self.left = left
        self.right = right


class Or:
    """Represents disjunction: φ ∨ ψ."""
    
    def __init__(self, left: "Formula", right: "Formula") -> None:
        self.left = left
        self.right = right


class Implies:
    """Represents implication: φ -> ψ."""
    
    def __init__(self, left: "Formula", right: "Formula") -> None:
        self.left = left
        self.right = right


class Forall:
    """Represents universal quantification: ∀x. φ."""
    
    def __init__(self, var: str, body: "Formula") -> None:
        self.var = var
        self.body = body


class Exists:
    """Represents existential quantification: ∃x. φ."""
    
    def __init__(self, var: str, body: "Formula") -> None:
        self.var = var
        self.body = body


Formula = Union[Predicate, Truth, Falsehood, Not, And, Or, Implies, Forall, Exists]


# ============================================================================
# Tokenizer
# ============================================================================

def tokenise(text: str) -> List[str]:
    """
    Tokenizes input text into a list of tokens.
    
    Supports:
    - Identifiers: P, Q, x, y, etc.
    - Operators: ->, ∧, ∨, ¬, &, |, ~, !
    - Quantifiers: ∀, ∃, forall, exists
    - Constants: ⊤, ⊥
    - Punctuation: (, ), ., ,
    
    Args:
        text: Input formula string
        
    Returns:
        List of tokens
        
    Raises:
        ParserError: If unexpected character encountered
    """
    tokens: List[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if text.startswith("->", index) or text.startswith("=>", index):
            # accept both '->' and '=>' as implication
            tokens.append("->")
            index += 2
            continue
        if text.startswith("<=>", index):
            tokens.append("<=>")
            index += 3
            continue
        if char in "()[],:&|¬∀∃⊤⊥~!?∧∨.":
            # include square brackets and colon used by TPTP quantifier syntax
            tokens.append(char)
            index += 1
            continue
        if char.isalpha() or char == "_":
            end = index + 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        raise ParserError(f"Unexpected character {char!r} in input")
    return tokens


# ============================================================================
# Parser
# ============================================================================

class FormulaParser:
    """
    Recursive descent parser for first-order logic formulae.
    
    Grammar (operator precedence from lowest to highest):
    - formula := implication
    - implication := or (-> implication)*
    - or := and ((| | ∨) and)*
    - and := unary ((& | ∧) unary)*
    - unary := (¬ | ~ | not) unary | quantifier | atom
    - quantifier := (∀ | forall) var . unary | (∃ | exists) var . unary
    - atom := (formula) | ⊤ | ⊥ | predicate(termlist) | predicate
    - termlist := term (, term)*
    - term := name (( termlist ))?
    """
    
    def __init__(self, tokens: Sequence[str]):
        self.tokens = list(tokens)
        self.position = 0

    def parse(self) -> Formula:
        """Parse the token list into a formula."""
        formula = self.parse_equivalence()
        if self.position != len(self.tokens):
            raise ParserError(f"Unexpected token {self.peek()!r}")
        return formula

    def peek(self) -> Optional[str]:
        """Look at the current token without consuming it."""
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def consume(self, expected: Optional[str] = None) -> str:
        """Consume and return the current token."""
        token = self.peek()
        if token is None:
            raise ParserError("Unexpected end of input")
        if expected is not None and token != expected:
            raise ParserError(f"Expected {expected!r}, got {token!r}")
        self.position += 1
        return token

    def parse_equivalence(self) -> Formula:
        """Parse equivalence (<=>) into an and of two implications."""
        left = self.parse_implication()
        while self.peek() == "<=>":
            self.consume("<=>")
            right = self.parse_implication()
            left = And(Implies(left, right), Implies(right, left))
        return left

    def parse_implication(self) -> Formula:
        """Parse implication (right-associative)."""
        left = self.parse_or()
        if self.peek() == "->":
            self.consume("->")
            right = self.parse_implication()
            return Implies(left, right)
        return left

    def parse_or(self) -> Formula:
        """Parse disjunction (left-associative)."""
        left = self.parse_and()
        while self.peek() in {"|", "∨", "or"}:
            self.consume()
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Formula:
        """Parse conjunction (left-associative)."""
        left = self.parse_unary()
        while self.peek() in {"&", "∧", "and"}:
            self.consume()
            right = self.parse_unary()
            left = And(left, right)
        return left

    def parse_unary(self) -> Formula:
        """Parse unary operators (negation, quantifiers) and atoms."""
        token = self.peek()
        if token in {"¬", "~", "not"}:
            self.consume()
            return Not(self.parse_unary())
        if token in {"∀", "forall"}:
            self.consume()
            var = self.consume_identifier("Expected variable after universal quantifier")
            if self.peek() == ".":
                self.consume(".")
            return Forall(var, self.parse_unary())
        if token in {"∃", "exists"}:
            self.consume()
            var = self.consume_identifier("Expected variable after existential quantifier")
            if self.peek() == ".":
                self.consume(".")
            return Exists(var, self.parse_unary())
        # TPTP style quantifiers: ![X,Y] : phi  and ?[X] : phi
        if token == "!" or token == "?":
            is_forall = token == "!"
            self.consume()
            if self.peek() != "[":
                raise ParserError("Expected '[' after quantifier")
            self.consume("[")
            vars: List[str] = []
            while True:
                v = self.consume_identifier("Expected variable name in quantifier list")
                vars.append(v)
                if self.peek() == ",":
                    self.consume(",")
                    continue
                break
            if self.peek() != "]":
                raise ParserError("Expected ']' after quantifier variable list")
            self.consume("]")
            if self.peek() == ":":
                self.consume(":")
            body = self.parse_unary()
            # nest quantifiers for multiple variables
            for v in reversed(vars):
                body = Forall(v, body) if is_forall else Exists(v, body)
            return body
        return self.parse_atom()

    def parse_atom(self) -> Formula:
        """Parse atomic formulas (predicates, constants, or parenthesized formulas)."""
        token = self.peek()
        if token == "(":
            self.consume("(")
            formula = self.parse_equivalence()
            self.consume(")")
            return formula
        if token == "⊤":
            self.consume("⊤")
            return Truth()
        if token == "⊥":
            self.consume("⊥")
            return Falsehood()

        name = self.consume_identifier("Expected predicate or term name")
        if self.peek() != "(":
            return Predicate(name)

        self.consume("(")
        args: List[Term] = []
        if self.peek() != ")":
            while True:
                args.append(self.parse_term())
                if self.peek() != ",":
                    break
                self.consume(",")
        self.consume(")")
        return Predicate(name, tuple(args))

    def parse_term(self) -> Term:
        """Parse a term (variable or function application)."""
        name = self.consume_identifier("Expected term name")
        if self.peek() != "(":
            return Term(name)

        self.consume("(")
        args: List[Term] = []
        if self.peek() != ")":
            while True:
                args.append(self.parse_term())
                if self.peek() != ",":
                    break
                self.consume(",")
        self.consume(")")
        return Term(name, tuple(args))

    def consume_identifier(self, message: str) -> str:
        """Consume and return an identifier token."""
        token = self.peek()
        if token is None:
            raise ParserError(message)
        if token in {"(", ")", ",", "&", "|", "->", "¬", "∀", "∃", "⊤", "⊥", "~", "!"}:
            raise ParserError(message)
        self.consume()
        return token


def parse_formula(text: str) -> Formula:
    """
    Parse a formula string into an AST.
    
    Args:
        text: Formula string
        
    Returns:
        Formula AST
        
    Raises:
        ParserError: If parsing fails
    """
    tokens = tokenise(text)
    try:
        return FormulaParser(tokens).parse()
    except ParserError:
        if tokens and tokens[-1] == ")":
            return FormulaParser(tokens[:-1]).parse()
        raise


# ============================================================================
# File I/O Functions
# ============================================================================

def readFormulas(filename: str) -> List[str]:
    """Read all lines from a formula file."""
    with open(filename, "r", encoding="utf-8") as handle:
        # If TPTP .p file, extract the formula argument from fof/cnf statements
        if filename.lower().endswith(".p") or filename.lower().endswith(".tptp"):
            return read_tptp_file(filename)
        with open(filename, "r", encoding="utf-8") as handle:
            return handle.readlines()


def read_tptp_file(filename: str) -> List[str]:
    """Extract formula texts from a TPTP file (FOF/CNF entries).

    This function looks for `fof(name, role, formula).` and `cnf(...)` lines
    and returns the formula component as strings ready for parsing.
    """
    import re

    formulas: List[str] = []
    pattern = re.compile(r"(?:fof|cnf)\s*\(\s*[^,]+\s*,\s*[^,]+\s*,\s*(.*)\)\s*\.\s*$", re.IGNORECASE)
    with open(filename, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            m = pattern.search(line)
            if m:
                phi = m.group(1).strip()
                # remove trailing period if present
                if phi.endswith("."):
                    phi = phi[:-1].strip()
                formulas.append(phi)
            else:
                # fallback: attempt to strip final period and take the line
                if line.endswith('.'):
                    formulas.append(line[:-1].strip())
                else:
                    formulas.append(line)
    return formulas


def read_tptp_entries(filename: str) -> List[Tuple[str, str]]:
    """Read TPTP fof/cnf entries as (role, formula_text) pairs.

    This parser is role-aware and multi-line safe for standard TPTP files.
    """
    import re

    entries: List[Tuple[str, str]] = []
    current: List[str] = []
    pattern = re.compile(r"^(?:fof|cnf)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(.*)\)\s*\.\s*$", re.IGNORECASE)

    with open(filename, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not current and not (line.startswith("fof(") or line.startswith("cnf(")):
                continue

            current.append(line)
            if not line.endswith(")."):
                continue

            block = " ".join(current)
            current = []
            match = pattern.match(block)
            if match:
                role = match.group(2).strip().lower()
                formula_text = match.group(3).strip()
                entries.append((role, formula_text))

    return entries


def build_tptp_problem_formula(entries: Sequence[Tuple[str, str]]) -> Optional[Formula]:
    """Convert one TPTP problem into a single formula for the prover.

    The baseline prover expects one formula, so we encode all axioms as a
    conjunction in the antecedent and the conjecture as the consequent.
    Axiom-only files are skipped because they are not proof goals.
    """
    axioms: List[Formula] = []
    conjectures: List[Formula] = []
    for role, formula_text in entries:
        if role == "axiom":
            axioms.append(parse_formula(formula_text))
        elif role == "conjecture":
            conjectures.append(parse_formula(formula_text))

    if not conjectures:
        return None

    conjecture = conjectures[-1]
    if not axioms:
        return conjecture

    antecedent = axioms[0]
    for axiom in axioms[1:]:
        antecedent = And(antecedent, axiom)
    return Implies(antecedent, conjecture)


def load_tptp_problems_from_folder(folder: str) -> List[Formula]:
    """Load one proof goal per TPTP file from a folder.

    Files without a conjecture are skipped.
    """
    import os

    problems: List[Formula] = []
    for entry in sorted(os.listdir(folder)):
        if not entry.lower().endswith(".p"):
            continue
        path = os.path.join(folder, entry)
        problem = build_tptp_problem_formula(read_tptp_entries(path))
        if problem is not None:
            problems.append(problem)
    return problems


def load_tptp_problems_from_path(path: str) -> List[Formula]:
    """Load TPTP proof goals from either a single file or a folder."""
    import os

    if os.path.isdir(path):
        return load_tptp_problems_from_folder(path)
    if path.lower().endswith((".p", ".tptp")):
        problem = build_tptp_problem_formula(read_tptp_entries(path))
        return [] if problem is None else [problem]
    return read_and_parse_formulas(path)


def stripFormulas(lines: Iterable[str]) -> List[str]:
    """Strip and clean formula strings, removing empty lines."""
    formulas: List[str] = []
    for line in lines:
        cleaned = line.strip().replace(" ", "")
        if cleaned:
            formulas.append(cleaned)
    return formulas


def read_and_parse_formulas(filename: str) -> List[Formula]:
    """Read and parse all formulas from a file."""
    return [parse_formula(formula) for formula in stripFormulas(readFormulas(filename))]


# ============================================================================
# String Conversion (Output Format)
# ============================================================================

def term_to_string(term: Term) -> str:
    """Convert a term to string representation."""
    if not term.args:
        return term.name
    return f"{term.name}({', '.join(term_to_string(arg) for arg in term.args)})"


def formula_to_string(formula: Formula) -> str:
    """
    Convert a formula to string representation.
    
    Output format uses Unicode symbols for better readability.
    """
    if isinstance(formula, Predicate):
        if not formula.args:
            return formula.name
        return f"{formula.name}({', '.join(term_to_string(arg) for arg in formula.args)})"
    if isinstance(formula, Truth):
        return "⊤"
    if isinstance(formula, Falsehood):
        return "⊥"
    if isinstance(formula, Not):
        return f"¬{formula_to_string(formula.operand)}"
    if isinstance(formula, And):
        return f"({formula_to_string(formula.left)} ∧ {formula_to_string(formula.right)})"
    if isinstance(formula, Or):
        return f"({formula_to_string(formula.left)} ∨ {formula_to_string(formula.right)})"
    if isinstance(formula, Implies):
        return f"({formula_to_string(formula.left)} -> {formula_to_string(formula.right)})"
    if isinstance(formula, Forall):
        return f"∀{formula.var}. {formula_to_string(formula.body)}"
    if isinstance(formula, Exists):
        return f"∃{formula.var}. {formula_to_string(formula.body)}"
    raise TypeError(f"Unsupported formula type: {type(formula)!r}")


# ============================================================================
# Substitution and Unification
# ============================================================================

def substitute_term(term: Term, var: str, replacement: Term) -> Term:
    """
    Substitute a variable with a term in a term.
    
    Args:
        term: The term to substitute in
        var: The variable name to substitute
        replacement: The term to replace it with
        
    Returns:
        New term with substitution applied
    """
    if term.name == var and not term.args:
        return replacement
    return Term(term.name, tuple(substitute_term(arg, var, replacement) for arg in term.args))


def substitute_formula(formula: Formula, var: str, replacement: Term) -> Formula:
    """
    Substitute a variable with a term throughout a formula.
    
    Args:
        formula: The formula to substitute in
        var: The variable name to substitute
        replacement: The term to replace it with
        
    Returns:
        New formula with substitution applied
    """
    if isinstance(formula, Predicate):
        return Predicate(formula.name, tuple(substitute_term(arg, var, replacement) for arg in formula.args))
    if isinstance(formula, Truth) or isinstance(formula, Falsehood):
        return formula
    if isinstance(formula, Not):
        return Not(substitute_formula(formula.operand, var, replacement))
    if isinstance(formula, And):
        return And(
            substitute_formula(formula.left, var, replacement),
            substitute_formula(formula.right, var, replacement),
        )
    if isinstance(formula, Or):
        return Or(
            substitute_formula(formula.left, var, replacement),
            substitute_formula(formula.right, var, replacement),
        )
    if isinstance(formula, Implies):
        return Implies(
            substitute_formula(formula.left, var, replacement),
            substitute_formula(formula.right, var, replacement),
        )
    if isinstance(formula, Forall):
        if formula.var == var:
            return formula
        return Forall(formula.var, substitute_formula(formula.body, var, replacement))
    if isinstance(formula, Exists):
        if formula.var == var:
            return formula
        return Exists(formula.var, substitute_formula(formula.body, var, replacement))
    raise TypeError(f"Unsupported formula type: {type(formula)!r}")


def is_variable_name(name: str) -> bool:
    """Check if a name is a variable (heuristic: names starting with 't' are Skolem/fresh constants)."""
    return not name.startswith("t")


def apply_subst_term(term: Term, subst: dict) -> Term:
    """Apply a substitution mapping to a term."""
    if not term.args and term.name in subst:
        return subst[term.name]
    return Term(term.name, tuple(apply_subst_term(arg, subst) for arg in term.args))


def apply_subst_formula(formula: Formula, subst: dict) -> Formula:
    """Apply a substitution mapping to a formula."""
    if isinstance(formula, Predicate):
        return Predicate(formula.name, tuple(apply_subst_term(arg, subst) for arg in formula.args))
    if isinstance(formula, Truth) or isinstance(formula, Falsehood):
        return formula
    if isinstance(formula, Not):
        return Not(apply_subst_formula(formula.operand, subst))
    if isinstance(formula, And):
        return And(apply_subst_formula(formula.left, subst), apply_subst_formula(formula.right, subst))
    if isinstance(formula, Or):
        return Or(apply_subst_formula(formula.left, subst), apply_subst_formula(formula.right, subst))
    if isinstance(formula, Implies):
        return Implies(apply_subst_formula(formula.left, subst), apply_subst_formula(formula.right, subst))
    if isinstance(formula, Forall):
        if formula.var in subst:
            new_subst = {k: v for k, v in subst.items() if k != formula.var}
            return Forall(formula.var, apply_subst_formula(formula.body, new_subst))
        return Forall(formula.var, apply_subst_formula(formula.body, subst))
    if isinstance(formula, Exists):
        if formula.var in subst:
            new_subst = {k: v for k, v in subst.items() if k != formula.var}
            return Exists(formula.var, apply_subst_formula(formula.body, new_subst))
        return Exists(formula.var, apply_subst_formula(formula.body, subst))
    raise TypeError(f"Unsupported formula type: {type(formula)!r}")


def unify_terms(t1: Term, t2: Term, subst: dict, var_names: Set[str]) -> Optional[dict]:
    """
    Unify two terms.
    
    Returns a substitution that makes the terms equal, or None if unification fails.
    """
    if t1.args or t2.args:
        if t1.name != t2.name or len(t1.args) != len(t2.args):
            return None
        new_subst = dict(subst)
        for a, b in zip(t1.args, t2.args):
            res = unify_terms(apply_subst_term(a, new_subst), apply_subst_term(b, new_subst), new_subst, var_names)
            if res is None:
                return None
            new_subst = res
        return new_subst

    a = apply_subst_term(t1, subst)
    b = apply_subst_term(t2, subst)
    if a.name == b.name and not a.args and not b.args:
        return dict(subst)

    if a.name in var_names:
        if occurs_in(a.name, b):
            return None
        new_subst = dict(subst)
        new_subst[a.name] = b
        return new_subst
    if b.name in var_names:
        if occurs_in(b.name, a):
            return None
        new_subst = dict(subst)
        new_subst[b.name] = a
        return new_subst
    return None


def occurs_in(var: str, term: Term) -> bool:
    """Check if a variable occurs in a term (occurs check for unification)."""
    if term.name == var and not term.args:
        return True
    for arg in term.args:
        if occurs_in(var, arg):
            return True
    return False


def unify_predicates(p1: Predicate, p2: Predicate, var_names: Set[str]) -> Optional[dict]:
    """Unify two predicates by their arguments."""
    if p1.name != p2.name or len(p1.args) != len(p2.args):
        return None
    subst: dict = {}
    for a, b in zip(p1.args, p2.args):
        res = unify_terms(a, b, subst, var_names)
        if res is None:
            return None
        subst = res
    return subst


# ============================================================================
# Term/Formula Collection
# ============================================================================

def collect_terms_from_term(term: Term) -> Set[str]:
    """Collect all term names in a term."""
    terms = {term.name}
    for arg in term.args:
        terms.update(collect_terms_from_term(arg))
    return terms


def collect_terms_from_formula(formula: Formula) -> Set[str]:
    """Collect all term and variable names in a formula."""
    if isinstance(formula, Predicate):
        terms: Set[str] = set()
        for arg in formula.args:
            terms.update(collect_terms_from_term(arg))
        return terms
    if isinstance(formula, Truth) or isinstance(formula, Falsehood):
        return set()
    if isinstance(formula, Not):
        return collect_terms_from_formula(formula.operand)
    if isinstance(formula, And) or isinstance(formula, Or) or isinstance(formula, Implies):
        return collect_terms_from_formula(formula.left) | collect_terms_from_formula(formula.right)
    if isinstance(formula, Forall) or isinstance(formula, Exists):
        return collect_terms_from_formula(formula.body) | {formula.var}
    raise TypeError(f"Unsupported formula type: {type(formula)!r}")


# ============================================================================
# Branch State for Sequent Calculus
# ============================================================================

class BranchState:
    """
    Represents the state of a branch in the sequent calculus proof.
    
    A branch contains:
    - left: list of formulae on the left side of the sequent (hypotheses)
    - right: list of formulae on the right side of the sequent (conclusions)
    - terms: set of all available term names for instantiation
    - next_fresh: counter for generating fresh variables
    - closed: whether the branch has been closed (proof found)
    """
    
    def __init__(
        self,
        left: List[Formula],
        right: List[Formula],
        terms: Optional[Set[str]] = None,
        next_fresh: int = 0,
        closed: bool = False,
    ) -> None:
        self.left = left
        self.right = right
        self.terms = set() if terms is None else terms
        self.next_fresh = next_fresh
        self.closed = closed


def initial_branch(formula: Formula) -> BranchState:
    """Create the initial branch for proving a formula."""
    return BranchState(left=[], right=[formula], terms=collect_terms_from_formula(formula))


def fresh_term(branch: BranchState) -> Term:
    """Generate a fresh term name not already in the branch."""
    while True:
        candidate = f"t{branch.next_fresh}"
        branch.next_fresh += 1
        if candidate not in branch.terms:
            branch.terms.add(candidate)
            return Term(candidate)


def is_atomic(formula: Formula) -> bool:
    """Check if a formula is atomic (no connectives)."""
    return isinstance(formula, Predicate) or isinstance(formula, Truth) or isinstance(formula, Falsehood)


def branch_closes(branch: BranchState) -> bool:
    """
    Check if a branch is closed (axiom found).
    
    A branch closes if:
    1. ⊤ appears on the right (idR rule)
    2. ⊥ appears on the left (idL rule)
    3. A predicate appears on both sides (can be unified)
    """
    if any(isinstance(formula, Truth) for formula in branch.right):
        return True
    if any(isinstance(formula, Falsehood) for formula in branch.left):
        return True

    right_atoms = [f for f in branch.right if is_atomic(f)]
    left_atoms = [f for f in branch.left if is_atomic(f)]
    right_set = {formula_to_string(f) for f in right_atoms}
    left_set = {formula_to_string(f) for f in left_atoms}
    if right_set & left_set:
        return True

    # Try unification-based closure
    var_names = {name for name in branch.terms if is_variable_name(name)}
    for L in left_atoms:
        if not isinstance(L, Predicate):
            continue
        for R in right_atoms:
            if not isinstance(R, Predicate):
                continue
            subst = unify_predicates(L, R, var_names)
            if subst is not None:
                branch.left = [apply_subst_formula(f, subst) for f in branch.left]
                branch.right = [apply_subst_formula(f, subst) for f in branch.right]
                for v in subst.values():
                    branch.terms.update(collect_terms_from_term(v))
                return True
    return False


def branch_closes_fast(branch: BranchState) -> bool:
    """Faster closure test used by the improved algorithm."""
    for formula in branch.right:
        if isinstance(formula, Truth):
            return True
    for formula in branch.left:
        if isinstance(formula, Falsehood):
            return True

    left_predicates = [formula for formula in branch.left if isinstance(formula, Predicate)]
    if not left_predicates:
        return False

    right_by_shape: dict[Tuple[str, int], List[Predicate]] = {}
    for formula in branch.right:
        if not isinstance(formula, Predicate):
            continue
        key = (formula.name, len(formula.args))
        if key not in right_by_shape:
            right_by_shape[key] = []
        right_by_shape[key].append(formula)

    if not right_by_shape:
        return False

    var_names = {name for name in branch.terms if is_variable_name(name)}
    for left_predicate in left_predicates:
        candidates = right_by_shape.get((left_predicate.name, len(left_predicate.args)), [])
        for right_predicate in candidates:
            subst = unify_predicates(left_predicate, right_predicate, var_names)
            if subst is not None:
                branch.left = [apply_subst_formula(f, subst) for f in branch.left]
                branch.right = [apply_subst_formula(f, subst) for f in branch.right]
                for value in subst.values():
                    branch.terms.update(collect_terms_from_term(value))
                return True
    return False


# ============================================================================
# Inference Rules for Algorithm 2
# ============================================================================

def apply_non_branching_rule(branch: BranchState) -> Optional[BranchState]:
    """
    Apply non-branching rules from the sequent calculus.
    
    Non-branching rules:
    - ∧L: A ∧ B on the left → add both A and B to left
    - ∨R: A ∨ B on the right → add both A and B to right
    - →R: A → B on the right → add A to left, B to right
    - ¬L: ¬A on the left → add A to right
    - ¬R: ¬A on the right → add A to left
    - ∀R: ∀x. A on the right → instantiate with fresh term
    - ∃L: ∃x. A on the left → instantiate with fresh term
    """
    if branch.left:
        formula = branch.left[-1]
        if isinstance(formula, And):
            return BranchState(
                left=branch.left[:-1] + [formula.left, formula.right],
                right=branch.right[:],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Not):
            return BranchState(
                left=branch.left[:-1],
                right=branch.right[:] + [formula.operand],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Exists):
            witness = fresh_term(branch)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            return BranchState(
                left=branch.left[:-1] + [instantiated],
                right=branch.right[:],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
    if branch.right:
        formula = branch.right[-1]
        if isinstance(formula, Or):
            return BranchState(
                left=branch.left[:],
                right=branch.right[:-1] + [formula.left, formula.right],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Implies):
            return BranchState(
                left=branch.left[:] + [formula.left],
                right=branch.right[:-1] + [formula.right],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Not):
            return BranchState(
                left=branch.left[:] + [formula.operand],
                right=branch.right[:-1],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Forall):
            witness = fresh_term(branch)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            return BranchState(
                left=branch.left[:],
                right=branch.right[:-1] + [instantiated],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Exists):
            witness = fresh_term(branch)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            return BranchState(
                left=branch.left[:],
                right=branch.right[:-1] + [instantiated],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )

    return None


def apply_branching_rule(branch: BranchState) -> Optional[List[BranchState]]:
    """
    Apply branching rules from the sequent calculus.
    
    Branching rules:
    - ∧R: A ∧ B on the right → create branches for A and B
    - ∨L: A ∨ B on the left → create branches for A and B
    - →L: A → B on the left → create branches for ¬A and B
    """
    if branch.left:
        formula = branch.left[-1]
        if isinstance(formula, Or):
            return [
                BranchState(
                    left=branch.left[:-1] + [formula.left],
                    right=branch.right[:],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
                BranchState(
                    left=branch.left[:-1] + [formula.right],
                    right=branch.right[:],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
            ]
        if isinstance(formula, Implies):
            return [
                BranchState(
                    left=branch.left[:-1],
                    right=branch.right[:] + [formula.left],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
                BranchState(
                    left=branch.left[:-1] + [formula.right],
                    right=branch.right[:],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
            ]

    if branch.right:
        formula = branch.right[-1]
        if isinstance(formula, And):
            return [
                BranchState(
                    left=branch.left[:],
                    right=branch.right[:-1] + [formula.left],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
                BranchState(
                    left=branch.left[:],
                    right=branch.right[:-1] + [formula.right],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
            ]

    return None


def apply_quantifier_instantiation(branch: BranchState) -> Optional[List[BranchState]]:
    """
    Apply quantifier instantiation rules.
    
    For ∀L and ∃R:
    - First, try all existing terms in the branch
    - Then, create a new branch with a fresh term
    """
    candidate_names = sorted(branch.terms, key=lambda n: n.startswith('t'))

    if branch.left and isinstance(branch.left[-1], Forall):
        formula = branch.left[-1]
        branches: List[BranchState] = []
        for name in candidate_names:
            witness = Term(name)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            branches.append(
                BranchState(
                    left=branch.left[:-1] + [instantiated],
                    right=branch.right[:],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                )
            )
        witness = fresh_term(branch)
        instantiated = substitute_formula(formula.body, formula.var, witness)
        branches.append(
            BranchState(
                left=branch.left[:-1] + [instantiated],
                right=branch.right[:],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        )
        return branches

    if branch.right and isinstance(branch.right[-1], Exists):
        formula = branch.right[-1]
        branches = []
        for name in candidate_names:
            witness = Term(name)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            branches.append(
                BranchState(
                    left=branch.left[:],
                    right=branch.right[:-1] + [instantiated],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                )
            )
        witness = fresh_term(branch)
        instantiated = substitute_formula(formula.body, formula.var, witness)
        branches.append(
            BranchState(
                left=branch.left[:],
                right=branch.right[:-1] + [instantiated],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        )
        return branches

    return None


# ============================================================================
# Algorithm 2: Baseline Proof Search
# ============================================================================

def run_algorithm_2(formula: Formula, max_steps: int = 256) -> List[BranchState]:
    """
    Run Algorithm 2 on a formula.
    
    Algorithm 2 (simplified tableau with backtracking):
    1. Start with an initial branch containing the formula on the right
    2. For each open branch:
       a. If the branch is closed, mark it as closed
       b. Otherwise, apply rules in this order:
          - Closing rules (idL, idR, unification)
          - Non-branching rules
          - Branching rules
          - Quantifier instantiation
       c. If no rule applies, leave the branch open
    3. Return all final branches
    
    Args:
        formula: The formula to prove
        max_steps: Maximum number of proof steps (avoid infinite loops)
        
    Returns:
        List of all final branch states (open and closed)
    """
    open_branches: List[BranchState] = [initial_branch(formula)]
    finished_branches: List[BranchState] = []
    steps = 0

    while open_branches and steps < max_steps:
        branch = open_branches.pop()
        steps += 1

        if branch_closes(branch):
            branch.closed = True
            finished_branches.append(branch)
            continue

        updated = apply_non_branching_rule(branch)
        if updated is not None:
            open_branches.append(updated)
            continue

        branched = apply_branching_rule(branch)
        if branched is not None:
            open_branches.extend(branched)
            continue

        instantiated = apply_quantifier_instantiation(branch)
        if instantiated is not None:
            open_branches.extend(instantiated)
            continue

        finished_branches.append(branch)

    finished_branches.extend(open_branches)
    return finished_branches


_TERM_KEY_CACHE: dict[int, object] = {}
_FORMULA_KEY_CACHE: dict[int, object] = {}


def term_structural_key(term: Term) -> object:
    """Return a cached structural key for a term."""
    cached = _TERM_KEY_CACHE.get(id(term))
    if cached is not None:
        return cached
    key = ("term", term.name, tuple(term_structural_key(arg) for arg in term.args))
    _TERM_KEY_CACHE[id(term)] = key
    return key


def formula_structural_key(formula: Formula) -> object:
    """Return a cached structural key for a formula."""
    cached = _FORMULA_KEY_CACHE.get(id(formula))
    if cached is not None:
        return cached

    if isinstance(formula, Predicate):
        key = ("pred", formula.name, tuple(term_structural_key(arg) for arg in formula.args))
    elif isinstance(formula, Truth):
        key = ("truth",)
    elif isinstance(formula, Falsehood):
        key = ("falsehood",)
    elif isinstance(formula, Not):
        key = ("not", formula_structural_key(formula.operand))
    elif isinstance(formula, And):
        key = ("and", formula_structural_key(formula.left), formula_structural_key(formula.right))
    elif isinstance(formula, Or):
        key = ("or", formula_structural_key(formula.left), formula_structural_key(formula.right))
    elif isinstance(formula, Implies):
        key = ("implies", formula_structural_key(formula.left), formula_structural_key(formula.right))
    elif isinstance(formula, Forall):
        key = ("forall", formula.var, formula_structural_key(formula.body))
    elif isinstance(formula, Exists):
        key = ("exists", formula.var, formula_structural_key(formula.body))
    else:
        key = ("unknown", repr(formula))

    _FORMULA_KEY_CACHE[id(formula)] = key
    return key


def branch_signature(branch: BranchState) -> Tuple[Tuple[object, ...], Tuple[object, ...], Tuple[str, ...], int]:
    """Create a hashable signature for a branch so repeated states can be skipped."""
    left = tuple(formula_structural_key(formula) for formula in branch.left)
    right = tuple(formula_structural_key(formula) for formula in branch.right)
    terms = tuple(sorted(branch.terms))
    return left, right, terms, branch.next_fresh


def apply_non_branching_rule_at(branch: BranchState, side: str, index: int) -> Optional[BranchState]:
    """Apply a non-branching rule to a specific formula inside a branch."""
    if side == "left":
        formula = branch.left[index]
        if isinstance(formula, And):
            return BranchState(
                left=branch.left[:index] + [formula.left, formula.right] + branch.left[index + 1 :],
                right=branch.right[:],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Not):
            return BranchState(
                left=branch.left[:index] + branch.left[index + 1 :],
                right=branch.right[:] + [formula.operand],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Exists):
            witness = fresh_term(branch)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            return BranchState(
                left=branch.left[:index] + [instantiated] + branch.left[index + 1 :],
                right=branch.right[:],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )

    if side == "right":
        formula = branch.right[index]
        if isinstance(formula, Or):
            return BranchState(
                left=branch.left[:],
                right=branch.right[:index] + [formula.left, formula.right] + branch.right[index + 1 :],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Implies):
            return BranchState(
                left=branch.left[:] + [formula.left],
                right=branch.right[:index] + [formula.right] + branch.right[index + 1 :],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Not):
            return BranchState(
                left=branch.left[:] + [formula.operand],
                right=branch.right[:index] + branch.right[index + 1 :],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Forall):
            witness = fresh_term(branch)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            return BranchState(
                left=branch.left[:],
                right=branch.right[:index] + [instantiated] + branch.right[index + 1 :],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        if isinstance(formula, Exists):
            witness = fresh_term(branch)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            return BranchState(
                left=branch.left[:],
                right=branch.right[:index] + [instantiated] + branch.right[index + 1 :],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )

    return None


def apply_branching_rule_at(branch: BranchState, side: str, index: int) -> Optional[List[BranchState]]:
    """Apply a branching rule to a specific formula inside a branch."""
    if side == "left":
        formula = branch.left[index]
        if isinstance(formula, Or):
            return [
                BranchState(
                    left=branch.left[:index] + [formula.left] + branch.left[index + 1 :],
                    right=branch.right[:],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
                BranchState(
                    left=branch.left[:index] + [formula.right] + branch.left[index + 1 :],
                    right=branch.right[:],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
            ]
        if isinstance(formula, Implies):
            return [
                BranchState(
                    left=branch.left[:index] + branch.left[index + 1 :],
                    right=branch.right[:] + [formula.left],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
                BranchState(
                    left=branch.left[:index] + [formula.right] + branch.left[index + 1 :],
                    right=branch.right[:],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
            ]

    if side == "right":
        formula = branch.right[index]
        if isinstance(formula, And):
            return [
                BranchState(
                    left=branch.left[:],
                    right=branch.right[:index] + [formula.left] + branch.right[index + 1 :],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
                BranchState(
                    left=branch.left[:],
                    right=branch.right[:index] + [formula.right] + branch.right[index + 1 :],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                ),
            ]

    return None


def apply_quantifier_instantiation_at(branch: BranchState, side: str, index: int) -> Optional[List[BranchState]]:
    """Apply quantifier instantiation to a specific formula inside a branch."""
    candidate_names = sorted(branch.terms, key=lambda n: n.startswith("t"))

    if side == "left" and isinstance(branch.left[index], Forall):
        formula = branch.left[index]
        branches: List[BranchState] = []
        for name in candidate_names:
            witness = Term(name)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            branches.append(
                BranchState(
                    left=branch.left[:index] + [instantiated] + branch.left[index + 1 :],
                    right=branch.right[:],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                )
            )
        witness = fresh_term(branch)
        instantiated = substitute_formula(formula.body, formula.var, witness)
        branches.append(
            BranchState(
                left=branch.left[:index] + [instantiated] + branch.left[index + 1 :],
                right=branch.right[:],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        )
        return branches

    if side == "right" and isinstance(branch.right[index], Exists):
        formula = branch.right[index]
        branches = []
        for name in candidate_names:
            witness = Term(name)
            instantiated = substitute_formula(formula.body, formula.var, witness)
            branches.append(
                BranchState(
                    left=branch.left[:],
                    right=branch.right[:index] + [instantiated] + branch.right[index + 1 :],
                    terms=set(branch.terms),
                    next_fresh=branch.next_fresh,
                )
            )
        witness = fresh_term(branch)
        instantiated = substitute_formula(formula.body, formula.var, witness)
        branches.append(
            BranchState(
                left=branch.left[:],
                right=branch.right[:index] + [instantiated] + branch.right[index + 1 :],
                terms=set(branch.terms),
                next_fresh=branch.next_fresh,
            )
        )
        return branches

    return None


def choose_improved_rule(branch: BranchState) -> Optional[Tuple[str, str, int]]:
    """Pick the next rule location for the improved search.

    The improved search scans the whole sequent, so it can reduce formulas that are
    applicable even when they are not the last item in the list.
    """
    for side, formulas, rule_types in (
        ("left", branch.left, (And, Not, Exists)),
        ("right", branch.right, (Or, Implies, Not, Forall, Exists)),
    ):
        for index, formula in enumerate(formulas):
            if isinstance(formula, rule_types):
                return side, "nonbranching", index

    for side, formulas, rule_types in (
        ("left", branch.left, (Or, Implies)),
        ("right", branch.right, (And,)),
    ):
        for index, formula in enumerate(formulas):
            if isinstance(formula, rule_types):
                return side, "branching", index

    for side, formulas, rule_types in (
        ("left", branch.left, (Forall,)),
        ("right", branch.right, (Exists,)),
    ):
        for index, formula in enumerate(formulas):
            if isinstance(formula, rule_types):
                return side, "quantifier", index

    return None


def formula_node_count(formula: Formula) -> int:
    """Return a simple structural size estimate for a formula."""
    if isinstance(formula, (Predicate, Truth, Falsehood)):
        return 1
    if isinstance(formula, Not):
        return 1 + formula_node_count(formula.operand)
    if isinstance(formula, (And, Or, Implies)):
        return 1 + formula_node_count(formula.left) + formula_node_count(formula.right)
    if isinstance(formula, (Forall, Exists)):
        return 1 + formula_node_count(formula.body)
    return 1


def _run_algorithm_2_improved_core(formula: Formula, max_steps: int = 256) -> List[BranchState]:
    """Low-overhead improved search core used for complex formulas."""
    open_branches: List[BranchState] = [initial_branch(formula)]
    finished_branches: List[BranchState] = []
    steps = 0

    # Cache seen branch signatures to avoid re-exploring identical states
    seen: Set[Tuple] = set()

    def branch_score(b: BranchState) -> Tuple[int, int]:
        # Prefer branches with fewer formulas and smaller structural size
        size = sum(formula_node_count(f) for f in (b.left + b.right))
        return (len(b.left) + len(b.right), size)

    while open_branches and steps < max_steps:
        branch = open_branches.pop()
        steps += 1

        sig = branch_signature(branch)
        if sig in seen:
            continue
        seen.add(sig)

        if branch_closes_fast(branch):
            branch.closed = True
            finished_branches.append(branch)
            continue

        # Try targeted improved rule selection (scans whole sequent)
        choice = choose_improved_rule(branch)
        if choice is not None:
            side, rtype, index = choice
            if rtype == "nonbranching":
                updated = apply_non_branching_rule_at(branch, side, index)
                if updated is not None:
                    s = branch_signature(updated)
                    if s not in seen:
                        open_branches.append(updated)
                    continue
            elif rtype == "branching":
                branched = apply_branching_rule_at(branch, side, index)
                if branched is not None:
                    # filter duplicates and order by heuristic
                    newbs = [b for b in branched if branch_signature(b) not in seen]
                    newbs.sort(key=branch_score)
                    open_branches.extend(newbs)
                    continue
            elif rtype == "quantifier":
                instantiated = apply_quantifier_instantiation_at(branch, side, index)
                if instantiated is not None:
                    newbs = [b for b in instantiated if branch_signature(b) not in seen]
                    newbs.sort(key=branch_score)
                    open_branches.extend(newbs)
                    continue

        # Fallback to single-step rules at list ends (maintain compatibility)
        updated = apply_non_branching_rule(branch)
        if updated is not None:
            s = branch_signature(updated)
            if s not in seen:
                open_branches.append(updated)
            continue

        branched = apply_branching_rule(branch)
        if branched is not None:
            newbs = [b for b in branched if branch_signature(b) not in seen]
            newbs.sort(key=branch_score)
            open_branches.extend(newbs)
            continue

        instantiated = apply_quantifier_instantiation(branch)
        if instantiated is not None:
            newbs = [b for b in instantiated if branch_signature(b) not in seen]
            newbs.sort(key=branch_score)
            open_branches.extend(newbs)
            continue

        finished_branches.append(branch)

    finished_branches.extend(open_branches)
    return finished_branches


def run_algorithm_2_improved(formula: Formula, max_steps: int = 256) -> List[BranchState]:
    """Adaptive improved proof search.

    Small formulas are handled by the baseline routine (lower constant overhead),
    while larger formulas use a faster closure test in the improved core.
    """
    if formula_node_count(formula) <= 6:
        return run_algorithm_2(formula, max_steps=max_steps)
    return _run_algorithm_2_improved_core(formula, max_steps=max_steps)


def benchmark_algorithms(
    formulas: Sequence[Formula],
    repeat: int = 1,
    max_steps: int = 256,
) -> List[dict]:
    """Benchmark the baseline and improved algorithms on the same formulas."""
    results: List[dict] = []

    for formula in formulas:
        baseline_times: List[float] = []
        improved_times: List[float] = []
        baseline_result: List[BranchState] = []
        improved_result: List[BranchState] = []

        for _ in range(repeat):
            start = time.perf_counter()
            baseline_result = run_algorithm_2(formula, max_steps=max_steps)
            baseline_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            improved_result = run_algorithm_2_improved(formula, max_steps=max_steps)
            improved_times.append(time.perf_counter() - start)

        baseline_closed = sum(1 for branch in baseline_result if branch.closed)
        improved_closed = sum(1 for branch in improved_result if branch.closed)
        baseline_avg = mean(baseline_times)
        improved_avg = mean(improved_times)

        results.append(
            {
                "formula": formula,
                "formula_text": formula_to_string(formula),
                "baseline_avg": baseline_avg,
                "improved_avg": improved_avg,
                "baseline_closed": baseline_closed,
                "baseline_open": len(baseline_result) - baseline_closed,
                "improved_closed": improved_closed,
                "improved_open": len(improved_result) - improved_closed,
                "speedup": (baseline_avg / improved_avg) if improved_avg > 0 else float("inf"),
            }
        )

    return results


def print_benchmark_report(results: Sequence[dict]) -> None:
    """Print a readable benchmark report."""
    print("\n" + "=" * 80)
    print("Benchmark: Baseline vs Improved Algorithm")
    print("=" * 80)
    print(f"{'#':>2}  {'Baseline (ms)':>14}  {'Improved (ms)':>14}  {'Speedup':>8}  Result")
    for index, result in enumerate(results, start=1):
        baseline_ms = result["baseline_avg"] * 1000.0
        improved_ms = result["improved_avg"] * 1000.0
        speedup = result["speedup"]
        status = (
            "same"
            if (
                result["baseline_closed"] == result["improved_closed"]
                and result["baseline_open"] == result["improved_open"]
            )
            else "diff"
        )
        print(f"{index:>2}  {baseline_ms:14.3f}  {improved_ms:14.3f}  {speedup:8.2f}x  {status}")

    baseline_total = sum(result["baseline_avg"] for result in results)
    improved_total = sum(result["improved_avg"] for result in results)
    overall_speedup = (baseline_total / improved_total) if improved_total > 0 else float("inf")
    print("-" * 80)
    print(f"Average baseline time: {baseline_total * 1000.0:.3f} ms")
    print(f"Average improved time: {improved_total * 1000.0:.3f} ms")
    print(f"Overall speedup: {overall_speedup:.2f}x")


def benchmark_formulas_from_file(filename: str, repeat: int = 1, max_steps: int = 256) -> None:
    """Read formulas from a file and benchmark both algorithms."""
    formulas = load_formulas_from_path(filename)
    results = benchmark_algorithms(formulas, repeat=repeat, max_steps=max_steps)
    print_benchmark_report(results)


def parse_tptp_folder(folder: str) -> List[Formula]:
    """Parse all .p TPTP files in a folder and return parsed formulas.

    Useful when you have a `tptp_fof` folder containing many .p problems.
    """
    import os
    formulas: List[Formula] = []
    for entry in sorted(os.listdir(folder)):
        if entry.lower().endswith('.p'):
            path = os.path.join(folder, entry)
            for ftext in read_tptp_file(path):
                try:
                    formulas.append(parse_formula(ftext))
                except ParserError:
                    # skip unparsable entries but continue
                    continue
    return formulas


def load_formulas_from_path(path: str) -> List[Formula]:
    """Load formulas from either a plain text file or a TPTP folder/file path."""
    import os

    if os.path.isdir(path):
        return parse_tptp_folder(path)
    if path.lower().endswith((".p", ".tptp")):
        return [parse_formula(formula) for formula in read_tptp_file(path)]
    return read_and_parse_formulas(path)


# ============================================================================
# Main Program
# ============================================================================

def main() -> None:
    """Main entry point: read formulas from file and run Algorithm 2 proof search."""
    input_file = input("Please enter a formula file or TPTP folder: ").strip()
    parsed_formulas = load_formulas_from_path(input_file)

    print("=" * 80)
    print("Parsed Formulae:")
    print("=" * 80)
    for i, formula in enumerate(parsed_formulas, start=1):
        print(f"{i:2}. {formula_to_string(formula)}")

    print("\n" + "=" * 80)
    print("Algorithm 2 Baseline Proof Search:")
    print("=" * 80)
    for index, formula in enumerate(parsed_formulas, start=1):
        branches = run_algorithm_2(formula)
        print(f"\nFormula {index}: {formula_to_string(formula)}")
        
        closed_count = sum(1 for b in branches if b.closed)
        open_count = len(branches) - closed_count
        
        if closed_count > 0 and open_count == 0:
            print(f"  ✓ PROOF FOUND ({closed_count} closed branch{'es' if closed_count != 1 else ''})")
        elif open_count > 0:
            print(f"  ✗ NO PROOF ({open_count} open branch{'es' if open_count != 1 else ''}, {closed_count} closed)")
        else:
            print(f"  ? Unknown ({closed_count} closed)")
        
        for branch_index, branch in enumerate(branches, start=1):
            left = ", ".join(formula_to_string(item) for item in branch.left) or ""
            right = ", ".join(formula_to_string(item) for item in branch.right) or ""
            state = "closed" if branch.closed else "open"
            print(f"    Branch {branch_index} [{state}]: {left} |- {right}")

    benchmark_choice = input("\nBenchmark baseline vs improved algorithm on this file? [y/N]: ").strip().lower()
    if benchmark_choice in {"y", "yes"}:
        repeat_input = input("How many repeats per formula? [1]: ").strip()
        repeat = int(repeat_input) if repeat_input else 1
        results = benchmark_algorithms(parsed_formulas, repeat=repeat)
        print_benchmark_report(results)


if __name__ == "__main__":
    main()

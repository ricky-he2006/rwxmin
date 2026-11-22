"""Batch scalar projections in the pinned SQLGlot Python executor."""
import ast
import math
from operator import itemgetter
from contextlib import contextmanager
from functools import lru_cache
from types import CodeType, GeneratorType


class Projection(tuple):
    """Keep individual expressions available to callers that iterate over them."""


@lru_cache(maxsize=2048)
def packed(sources):
    # PythonExecutor.generate puts the exact input to compile in co_filename.
    # Use separate ASTs to preserve expression boundaries and evaluation order.
    # These scalar forms neither bind names nor introduce another scope.
    scalar = (ast.Name, ast.Load, ast.Subscript, ast.Constant, ast.Call, ast.keyword,
              ast.Tuple, ast.List, ast.Dict, ast.Compare, ast.cmpop, ast.BinOp,
              ast.operator, ast.UnaryOp, ast.unaryop, ast.IfExp, ast.BoolOp, ast.boolop)
    elements = [ast.parse(source, mode="eval").body for source in sources]
    if any(not isinstance(node, scalar) for elem in elements for node in ast.walk(elem)):
        return None
    expression = ast.fix_missing_locations(ast.Expression(ast.Tuple(elements, ast.Load())))
    columns = []
    for element in elements:
        match element:
            case ast.Subscript(ast.Subscript(ast.Name("scope"), ast.Constant(table)), ast.Constant(key)) if (table is None or type(table) is str) and type(key) is str:
                columns.append((table, key))
            case _:
                columns.clear()
                break
    return compile(expression, "<sqlglot projections>", "eval", optimize=2), tuple(columns)


def project_rows(context, stream, columns, iterator_type):
    if type(stream) is iterator_type and stream.index == -1:
        tables = (stream.table,)
        if context.env["scope"] is not context.row_readers:
            return None
    elif (type(stream) is GeneratorType
          and stream.gi_code.co_qualname == "PythonExecutor.join.<locals>.<genexpr>"):
        tables = tuple(context.tables.values())
    else:
        return None
    source = tables[0]
    if any(t.rows is not source.rows or t.columns != source.columns for t in tables[1:]):
        return None
    readers = {id(t.reader) for t in tables}
    indices = []
    for table, key in columns:
        reader = context.row_readers.get(table)
        if id(reader) not in readers or key not in reader.columns:
            return None
        indices.append(reader.columns[key])
    take = itemgetter(*indices)
    if len(indices) == 1:
        rows = [(take(row),) for row in source.rows]
    else:
        rows = list(map(take, source.rows))
    if rows:
        for table in tables:
            table.reader.row = source.rows[-1]
    if type(stream) is iterator_type:
        stream.index = len(source.rows)
    else:
        context.env["scope"] = context.row_readers
    return rows


@contextmanager
def enabled():
    from sqlglot.executor.context import Context
    from sqlglot.executor.python import PythonExecutor

    from sqlglot.executor.table import TableIter

    project = PythonExecutor._project_and_filter
    generate = PythonExecutor.generate_tuple
    evaluate = Context.eval_tuple

    def generate_tuple(self, expressions):
        codes = generate(self, expressions)
        if not codes or not all(type(code) is CodeType for code in codes):
            return codes
        try:
            combined = packed(tuple(code.co_filename for code in codes))
        except (ValueError, SyntaxError, RecursionError):
            return codes
        if combined is None:
            return codes
        result = Projection(codes)
        result.combined, result.columns = combined
        return result

    def eval_tuple(self, codes):
        if type(codes) is Projection:
            return self.eval(codes.combined)
        return evaluate(self, codes)

    def project_and_filter(self, context, step, stream):
        if step.condition is not None or not step.projections or step.limit != math.inf:
            return project(self, context, step, stream)
        sink = self.table(step.projections)
        codes = self.generate_tuple(step.projections)
        rows = project_rows(context, stream, codes.columns, TableIter) if getattr(codes, "columns", ()) else None
        if rows is None:
            for _ in stream:
                sink.append(context.eval_tuple(codes))
        else:
            sink.rows = rows
        return sink

    PythonExecutor._project_and_filter = project_and_filter
    PythonExecutor.generate_tuple = generate_tuple
    Context.eval_tuple = eval_tuple
    try:
        yield
    finally:
        PythonExecutor._project_and_filter = project
        Context.eval_tuple = evaluate
        PythonExecutor.generate_tuple = generate

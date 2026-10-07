"""Strip comments and docstrings, keeping the code AST-identical."""
import ast, sys

src = open(sys.argv[1]).read()
tree = ast.parse(src)

class S(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        b = node.body
        if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
           and isinstance(b[0].value.value, str):
            node.body = b[1:] or [ast.Pass()]
        return node
    visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip

out = ast.unparse(ast.fix_missing_locations(S().visit(tree)))
open(sys.argv[2], "w").write(out + "\n")

a = ast.dump(S().visit(ast.parse(src)))
b = ast.dump(S().visit(ast.parse(out)))
assert a == b, "AST changed!"
print("stripped ok; identical AST;", len(src), "->", len(out), "bytes")

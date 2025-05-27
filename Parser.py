# parser.py
import re
import numpy as np

def ler_modelo_matematico_txt(path):
    with open(path, 'r') as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    tipo_linha = linhas[0].lower()
    tipo = "min" if "min" in tipo_linha else "max"

    obj_expr = re.findall(r'[-+]?\s*\d*\s*x\d+', tipo_linha)
    variaveis = []
    c = []
    for termo in obj_expr:
        coef, var = termo.replace(" ", "").split("x")
        coef = coef.replace("+", "")
        coef = -1 if coef == "-" else (1 if coef == "" else float(coef))
        c.append(coef)
        variaveis.append(f"x{var}")

    if tipo == "max":
        c = [-ci for ci in c]

    restricoes = linhas[1:-1]
    A = []
    b = []
    folgas = []
    artificiais = []
    nomes_base = []
    nomes_nao_base = variaveis.copy()

    for i, r in enumerate(restricoes):
        coef_linha = [0.0] * len(variaveis)
        extra_cols = []

        if "<=" in r:
            lhs, rhs = r.split("<=")
            sinal = "<="
        elif ">=" in r:
            lhs, rhs = r.split(">=")
            sinal = ">="
        elif "=" in r:
            lhs, rhs = r.split("=")
            sinal = "="
        else:
            raise ValueError("Restrição mal formatada: " + r)

        rhs = float(rhs.strip())

        termos = re.findall(r'[-+]?\s*\d*\s*x\d+', lhs)
        for termo in termos:
            termo = termo.replace(" ", "")
            m = re.match(r'([-+]?\d*)(x\d+)', termo)
            coef = m.group(1)
            var = m.group(2)
            coef = coef.replace("+", "")
            coef = -1 if coef == "-" else (1 if coef == "" else float(coef))
            idx = variaveis.index(var)
            coef_linha[idx] = coef

        if rhs < 0:
            coef_linha = [-a for a in coef_linha]
            rhs = -rhs
            sinal = {"<=": ">=", ">=": "<="}.get(sinal, sinal)

        linha_idx = len(A)

        if sinal == "<=":
            nome = f"s{linha_idx+1}"
            nomes_base.append(nome)
            folgas.append(nome)
            extra_cols = [1.0 if j == linha_idx else 0.0 for j in range(len(restricoes))]
        elif sinal == ">=":
            nome_sobra = f"e{linha_idx+1}"
            nome_art = f"a{linha_idx+1}"
            nomes_base.append(nome_art)
            folgas.append(nome_sobra)
            artificiais.append(nome_art)
            excesso_col = [-1.0 if j == linha_idx else 0.0 for j in range(len(restricoes))]
            artificial_col = [1.0 if j == linha_idx else 0.0 for j in range(len(restricoes))]
            extra_cols = excesso_col + artificial_col
        elif sinal == "=":
            nome_art = f"a{linha_idx+1}"
            nomes_base.append(nome_art)
            artificiais.append(nome_art)
            artificial_col = [1.0 if j == linha_idx else 0.0 for j in range(len(restricoes))]
            extra_cols = artificial_col

        A.append(coef_linha + extra_cols)
        b.append(rhs)

    # Garante que todas as linhas tenham o mesmo número de colunas
    max_cols = max(len(row) for row in A)
    for i in range(len(A)):
        if len(A[i]) < max_cols:
            A[i] += [0.0] * (max_cols - len(A[i]))

    A = np.array(A, dtype=float)
    c_extendida = c + [0.0] * (A.shape[1] - len(c))

    return {
        "A": A,
        "b": np.array(b, dtype=float),
        "c": np.array(c_extendida, dtype=float),
        "nomes_base": nomes_base,
        "nomes_nao_base": nomes_nao_base,
        "variaveis_artificiais": artificiais,
        "tipo_original": tipo
    }

import re
import numpy as np

def ler_modelo_matematico_txt(path):
    """
    Lê um arquivo texto com problema de PL, retorna as estruturas para Simplex.
    """

    with open(path, 'r') as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    tipo_linha = linhas[0].lower()
    tipo = "min" if "min" in tipo_linha else "max"

    # Extrai termos da função objetivo (ex: 2x1, -x2)
    obj_expr = re.findall(r'[-+]?\s*\d*\s*x\d+', tipo_linha)
    variaveis = []
    c = []
    for termo in obj_expr:
        coef, var = termo.replace(" ", "").split("x")
        coef = coef.replace("+", "")
        coef = -1 if coef == "-" else (1 if coef == "" else float(coef))
        nome_var = f"x{var}"
        if nome_var not in variaveis:
            variaveis.append(nome_var)
            c.append(coef)

    # Inverte sinais se for max (para minimizar padrão)
    if tipo == "max":
        c = [-ci for ci in c]

    restricoes = linhas[1:-1]
    A = []
    b = []

    folgas = []           # nomes de variáveis de folga (slack)
    excessos = []         # nomes de variáveis de excesso (surplus)
    artificiais = []      # nomes das variáveis artificiais
    nomes_base = []       # nomes das variáveis na base inicial
    nomes_nao_base = variaveis.copy()  # inicialmente as originais não-base

    # Variáveis auxiliares (folga, excesso, artificiais) serão adicionadas aqui:
    aux_vars = []

    for i, r in enumerate(restricoes):
        coef_linha = [0.0] * len(variaveis)

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

        # Inverte sinais e desigualdades se RHS < 0
        if rhs < 0:
            coef_linha = [-a for a in coef_linha]
            rhs = -rhs
            sinal = {"<=": ">=", ">=": "<="}.get(sinal, sinal)

        # Agora cria colunas auxiliares para essa restrição
        row_extra = []

        if sinal == "<=":
            # Folga (+1)
            nome_folga = f"s{i+1}"
            folgas.append(nome_folga)
            nomes_base.append(nome_folga)
            aux_vars.append(nome_folga)
            # coluna com 1 no i-ésimo lugar para folga
            for j in range(len(restricoes)):
                row_extra.append(1.0 if j == i else 0.0)

        elif sinal == ">=":
            # Excesso (-1) + Artificial (+1)
            nome_excesso = f"e{i+1}"
            nome_artificial = f"a{i+1}"
            excessos.append(nome_excesso)
            artificiais.append(nome_artificial)
            nomes_base.append(nome_artificial)  # artificial na base inicial
            aux_vars.extend([nome_excesso, nome_artificial])

            # colunas: excesso = -1 no i, artificial = +1 no i
            excesso_col = []
            artificial_col = []
            for j in range(len(restricoes)):
                excesso_col.append(-1.0 if j == i else 0.0)
                artificial_col.append(1.0 if j == i else 0.0)
            row_extra.extend(excesso_col)
            row_extra.extend(artificial_col)

        elif sinal == "=":
            # Artificial (+1)
            nome_artificial = f"a{i+1}"
            artificiais.append(nome_artificial)
            nomes_base.append(nome_artificial)
            aux_vars.append(nome_artificial)

            for j in range(len(restricoes)):
                row_extra.append(1.0 if j == i else 0.0)

        # Adiciona a linha à matriz A
        A.append(coef_linha + row_extra)
        b.append(rhs)

    # Completa com zeros se alguma linha ficou menor que o número máximo de colunas
    max_cols = max(len(row) for row in A)
    for i in range(len(A)):
        if len(A[i]) < max_cols:
            A[i] += [0.0] * (max_cols - len(A[i]))

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Vetor de custos c estendido com zeros para variáveis auxiliares
    c_extendida = c + [0.0] * (A.shape[1] - len(c))

    # Agora monta lista completa de nomes de variáveis na ordem das colunas da A
    nomes_variaveis = variaveis + aux_vars

    return {
        "A": A,
        "b": b,
        "c": np.array(c_extendida, dtype=float),
        "nomes_base": nomes_base,
        "nomes_nao_base": nomes_nao_base,
        "variaveis_artificiais": artificiais,
        "tipo_original": tipo,
        "nomes_variaveis": nomes_variaveis  # importante para indexação no main.py
    }

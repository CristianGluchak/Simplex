import re
import numpy as np

def ler_modelo_matematico_txt(path):
    """
    Lê um arquivo de texto com a formulação de um problema de Programação Linear (PL) e retorna
    as estruturas necessárias para resolução pelo método Simplex (com suporte a Fase 1 e Fase 2).

    Espera uma estrutura como:
        Max f(x) = 2x1 + 3x2
        x1 + x2 <= 4
        x1 - x2 >= 2
        x1, x2 >= 0

    Parâmetros:
        path (str): Caminho para o arquivo de entrada.

    Retorna:
        dict contendo:
            - A (np.ndarray): matriz de coeficientes das restrições
            - b (np.ndarray): vetor dos termos independentes
            - c (np.ndarray): vetor da função objetivo estendido (inclui zeros para variáveis auxiliares)
            - nomes_base (list): nomes das variáveis inicialmente na base
            - nomes_nao_base (list): nomes das variáveis originais
            - variaveis_artificiais (list): nomes das variáveis artificiais
            - tipo_original (str): 'max' ou 'min'
    """
    with open(path, 'r') as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    # Identifica se o problema é de minimização ou maximização
    tipo_linha = linhas[0].lower()
    tipo = "min" if "min" in tipo_linha else "max"

    # Extrai os termos da função objetivo (ex: 2x1, -x2)
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

    # Inverte sinais se for problema de maximização (para usar minimização padrão no Simplex)
    if tipo == "max":
        c = [-ci for ci in c]

    # Processa as restrições (linhas entre a função objetivo e a linha final de variáveis >= 0)
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

        # Detecta tipo de desigualdade
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

        # Extrai coeficientes do lado esquerdo da restrição
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

        # Se RHS for negativo, inverte a desigualdade e os sinais
        if rhs < 0:
            coef_linha = [-a for a in coef_linha]
            rhs = -rhs
            sinal = {"<=": ">=", ">=": "<="}.get(sinal, sinal)

        linha_idx = len(A)

        # Adiciona variáveis auxiliares conforme o tipo de restrição
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

    # Ajusta o tamanho das linhas da matriz A para que todas tenham o mesmo número de colunas
    max_cols = max(len(row) for row in A)
    for i in range(len(A)):
        if len(A[i]) < max_cols:
            A[i] += [0.0] * (max_cols - len(A[i]))

    A = np.array(A, dtype=float)

    # Estende o vetor c (função objetivo) com zeros para as variáveis auxiliares
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

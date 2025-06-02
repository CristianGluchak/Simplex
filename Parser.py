import re
import numpy as np

def ler_modelo_matematico_txt(path):
    """
    Lê um problema de Programação Linear em formato textual e o transforma em estruturas de dados
    adequadas para o método Simplex, incluindo:
        - Matriz de restrições A
        - Vetor de constantes b
        - Função objetivo c (convertida para minimização)
        - Variáveis auxiliares (slack, excess, artificial)
        - Identificação da base inicial
        - Mapeamento dos nomes de variáveis (para manter rastreabilidade)
    
    A função lida com desigualdades e igualdades, gerando colunas auxiliares para
    manter a estrutura da matriz aumentada padrão para Simplex.
    """
    with open(path, 'r') as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    # Identifica se o problema é de minimização ou maximização
    tipo_linha = linhas[0].lower()
    tipo = "min" if "min" in tipo_linha else "max"

    # Extrai termos da função objetivo (ex: 2x1, -x2, etc)
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

    # Se for problema de maximização, converte os coeficientes para o padrão de minimização
    if tipo == "max":
        c = [-ci for ci in c]

    # Processa as restrições
    restricoes = linhas[1:-1]
    A = []
    b = []

    folgas = []           # Variáveis de folga (slack) para <=
    excessos = []         # Variáveis de excesso (surplus) para >=
    artificiais = []      # Variáveis artificiais (necessárias para >= ou =)
    nomes_base = []       # Nomes das variáveis inicialmente na base (normalmente as auxiliares)
    nomes_nao_base = variaveis.copy()  # x1, x2... inicialmente estão fora da base

    aux_vars = []  # Nome de todas variáveis adicionais a serem adicionadas à matriz A

    for i, r in enumerate(restricoes):
        coef_linha = [0.0] * len(variaveis)

        # Identifica o tipo da restrição
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

        # Coleta os coeficientes da LHS (lado esquerdo)
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

        # Se o lado direito for negativo, inverte a inequação e os coeficientes
        if rhs < 0:
            coef_linha = [-a for a in coef_linha]
            rhs = -rhs
            sinal = {"<=": ">=", ">=": "<="}.get(sinal, sinal)

        # Adiciona variáveis auxiliares conforme o tipo da restrição
        row_extra = []

        if sinal == "<=":
            # Adiciona variável de folga (slack) com coeficiente +1
            nome_folga = f"s{i+1}"
            folgas.append(nome_folga)
            nomes_base.append(nome_folga)
            aux_vars.append(nome_folga)
            for j in range(len(restricoes)):
                row_extra.append(1.0 if j == i else 0.0)

        elif sinal == ">=":
            # Adiciona variável de excesso (-1) e artificial (+1)
            nome_excesso = f"e{i+1}"
            nome_artificial = f"a{i+1}"
            excessos.append(nome_excesso)
            artificiais.append(nome_artificial)
            nomes_base.append(nome_artificial)
            aux_vars.extend([nome_excesso, nome_artificial])

            excesso_col = [-1.0 if j == i else 0.0 for j in range(len(restricoes))]
            artificial_col = [1.0 if j == i else 0.0 for j in range(len(restricoes))]
            row_extra.extend(excesso_col + artificial_col)

        elif sinal == "=":
            # Adiciona variável artificial (+1)
            nome_artificial = f"a{i+1}"
            artificiais.append(nome_artificial)
            nomes_base.append(nome_artificial)
            aux_vars.append(nome_artificial)
            for j in range(len(restricoes)):
                row_extra.append(1.0 if j == i else 0.0)

        # Monta a linha da matriz A
        A.append(coef_linha + row_extra)
        b.append(rhs)

    # Garante que todas as linhas tenham o mesmo número de colunas (preenchendo com zeros)
    max_cols = max(len(row) for row in A)
    for i in range(len(A)):
        if len(A[i]) < max_cols:
            A[i] += [0.0] * (max_cols - len(A[i]))

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Extende o vetor de custos c com zeros para todas as variáveis auxiliares
    c_extendida = c + [0.0] * (A.shape[1] - len(c))

    # Cria a lista completa de nomes de variáveis na ordem exata das colunas da matriz A
    nomes_variaveis = variaveis + aux_vars

    return {
        "A": A,
        "b": b,
        "c": np.array(c_extendida, dtype=float),
        "nomes_base": nomes_base,
        "nomes_nao_base": nomes_nao_base,
        "variaveis_artificiais": artificiais,
        "tipo_original": tipo,
        "nomes_variaveis": nomes_variaveis
    }

from Parser import ler_modelo_matematico_txt
from Simplex import simplex_fase2
import numpy as np

def construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais):
    """
    Constrói as matrizes B e N, além dos vetores cb e cn,
    necessários para executar a Fase 1 do método Simplex.
    
    A base inicial é composta pelas variáveis artificiais.
    """
    B = []
    N = []
    cb = []
    cn = []

    # Monta matriz da base B e vetor cb
    for i, nome in enumerate(nomes_base):
        B.append(A[:, i])
        cb.append(1.0 if nome in artificiais else 0.0)

    # Monta matriz não-base N e vetor cn (zerado na Fase 1)
    for i, nome in enumerate(nomes_nao_base):
        col_idx = len(nomes_base) + i
        N.append(A[:, col_idx])
        cn.append(0.0)

    B = np.column_stack(B)
    N = np.column_stack(N)

    return B, N, b, np.array(cn), nomes_base.copy(), nomes_nao_base.copy()

def main():
    # Lê e interpreta o modelo matemático a partir do arquivo de entrada
    dados = ler_modelo_matematico_txt("entrada.txt")
    A = dados["A"]
    b = dados["b"]
    c = dados["c"]
    nomes_base = dados["nomes_base"]
    nomes_nao_base = dados["nomes_nao_base"]
    artificiais = dados["variaveis_artificiais"]

    #  Caso existam variáveis artificiais, é necessário executar a Fase 1
    if artificiais:
        print(">>> FASE 1: resolvendo problema artificial...")

        # Constrói as estruturas da Fase 1
        B, N, tb, cn_f1, base, nao_base = construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais)

        # Executa Fase 1 com função objetivo auxiliar
        solucao_f1, valor_f1 = simplex_fase2(B, N, tb, cn_f1, base, nao_base)

        # Verifica se foi possível encontrar solução viável
        if solucao_f1 is None or valor_f1 > 1e-6:
            print("\n Problema inviavel.")
            return

        # Verifica se ainda existem variáveis artificiais na base com valor > 0
        for nome in base:
            if nome in artificiais and solucao_f1.get(nome, 0.0) > 1e-6:
                print(" Problema inviavel: variavel artificial ainda ativa na base com valor > 0.")
                return

        print("\n Solução viavel encontrada na Fase I.")
        print(">>> FASE 2: otimizando função original...")

        # Prepara para a Fase 2 com função objetivo original
        todas = base + nao_base
        cn = np.array([c[i] for i, nome in enumerate(todas) if nome in nao_base])
        simplex_fase2(B, N, b, cn, base, nao_base)

    else:
        print(">>> FASE 2 DIRETA")

        m = len(b)
        B = A[:, :m]  # Colunas iniciais da base

        try:
            # Tenta inverter a base
            np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("Base singular. Alternando para Fase 1...")

            # Cria variáveis artificiais fictícias
            artificiais = [f"a{i+1}" for i in range(m)]
            nomes_base = artificiais.copy()
            nomes_nao_base = [f"x{i+1}" for i in range(A.shape[1])]

            # Estende a matriz A e o vetor de custos c
            A = np.hstack([A, np.identity(m)])
            c = np.concatenate([c, np.ones(m)])

            # Redefine base e não-base
            B = A[:, -m:]
            N = A[:, :-m]
            cn_f1 = [0.0] * N.shape[1]

            # Executa Fase 1 com base artificial
            solucao_f1, valor_f1 = simplex_fase2(B, N, b, cn_f1, nomes_base, nomes_nao_base)

            # Verifica viabilidade da solução
            if solucao_f1 is None or valor_f1 > 1e-6:
                print("\n Problema inviavel.")
                return

            # Verifica se alguma artificial permaneceu ativa na base
            for nome in nomes_base:
                if nome.startswith('a') and solucao_f1.get(nome, 0.0) > 1e-6:
                    print(" Problema inviavel: variavel artificial ainda ativa apos Fase 1 forçada.")
                    return

            print("\n Solução viável obtida após Fase 1 forçada. Pronto para Fase 2.")

            # Prepara custo cn com base nos nomes
            todas = nomes_base + nomes_nao_base
            cn = np.array([c[i] for i, nome in enumerate(todas) if nome in nomes_nao_base])
            simplex_fase2(B, N, b, cn, nomes_base, nomes_nao_base)
            return

        # Fase 2 direta com base viável
        N = A[:, m:]
        cn = np.array([c[nomes_nao_base.index(nome)] if nome in nomes_nao_base else 0.0 for nome in nomes_nao_base])
        simplex_fase2(B, N, b, cn, nomes_base, nomes_nao_base)

# Executa o programa
if __name__ == "__main__":
    main()

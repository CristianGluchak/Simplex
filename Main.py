from Parser import ler_modelo_matematico_txt
from Simplex import simplex_fase2
import numpy as np

def construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais):
    """
    Constrói as matrizes B (base) e N (não base) para a Fase 1 do método Simplex.
    A base inicial é composta pelas variáveis artificiais (com custo 1), e as demais
    variáveis recebem custo 0 na função objetivo auxiliar.

    Retorna:
        B, N: matrizes com colunas da base e fora da base
        b: vetor do lado direito do sistema
        cn: custos das variáveis não básicas (tudo zero)
        nomes_base, nomes_nao_base: cópias atualizadas dos nomes
    """
    B_cols = []
    N_cols = []
    cb = []
    cn = []

    # Constrói a matriz da base com variáveis artificiais
    for nome in nomes_base:
        if nome in nomes_base:
            idx = nomes_base.index(nome)
        else:
            idx = nomes_nao_base.index(nome) + len(nomes_base)
        B_cols.append(A[:, idx])
        cb.append(1.0 if nome in artificiais else 0.0)  # custo 1 para artificiais

    # Constrói a matriz fora da base com custo zero
    for nome in nomes_nao_base:
        idx = nomes_nao_base.index(nome) + len(nomes_base)
        N_cols.append(A[:, idx])
        cn.append(0.0)

    B = np.column_stack(B_cols)
    N = np.column_stack(N_cols)
    cn = np.array(cn)

    return B, N, b, cn, nomes_base.copy(), nomes_nao_base.copy()

def main():
    # Lê os dados do modelo (função objetivo, restrições e variáveis)
    dados = ler_modelo_matematico_txt("entrada.txt")
    A = dados["A"]
    b = dados["b"]
    c = dados["c"]
    nomes_base = dados["nomes_base"]
    nomes_nao_base = dados["nomes_nao_base"]
    artificiais = dados["variaveis_artificiais"]
    nomes_variaveis = dados["nomes_variaveis"]  # nomes de todas as colunas na ordem de A

    if artificiais:
        print(">>> FASE 1: resolvendo problema artificial...")

        # Inicializa a Fase 1 com base artificial
        B, N, tb, cn_f1, base, nao_base = construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais)
        solucao_f1, valor_f1 = simplex_fase2(B, N, tb, cn_f1, base, nao_base, A)

        # Se a função auxiliar não foi zerada, o problema original é inviável
        if solucao_f1 is None or valor_f1 > 1e-6:
            print("\n Problema inviavel.")
            return

        # Se ainda há variável artificial ativa na base, problema também é inviável
        for nome in base:
            if nome in artificiais and solucao_f1.get(nome, 0.0) > 1e-6:
                print(" Problema inviavel: variavel artificial ainda ativa na base com valor > 0.")
                return

        print("\n Solução viavel encontrada na Fase I.")
        print(">>> FASE 2: otimizando função original...")

        # Reconstrói base e não base com nomes corretos para a Fase 2
        indices_base = [nomes_variaveis.index(nome) for nome in base]
        indices_nao_base = [nomes_variaveis.index(nome) for nome in nao_base]

        B = A[:, indices_base]
        N = A[:, indices_nao_base]
        cn = np.array([c[i] for i in indices_nao_base])  # custos reais para Fase 2

        # Executa Fase 2 com a base viável encontrada
        simplex_fase2(B, N, b, cn, base, nao_base, A)

    else:
        print(">>> FASE 2 DIRETA")

        # Caso não existam variáveis artificiais, inicia direto na Fase 2
        m = len(b)

        indices_base = [nomes_variaveis.index(nome) for nome in nomes_base]
        indices_nao_base = [nomes_variaveis.index(nome) for nome in nomes_nao_base]

        B = A[:, indices_base]
        N = A[:, indices_nao_base]

        try:
            # Verifica se a base inicial é invertível
            np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("Base singular. Alternando para Fase 1...")

            # Gera variáveis artificiais para todas as restrições
            artificiais = [f"a{i+1}" for i in range(m)]
            nomes_base = artificiais.copy()
            nomes_nao_base = [f"x{i+1}" for i in range(A.shape[1])]

            # Adiciona identidade para representar variáveis artificiais
            A = np.hstack([A, np.identity(m)])
            c = np.concatenate([c, np.ones(m)])  # custos 1 para artificiais

            indices_base = list(range(A.shape[1] - m, A.shape[1]))
            indices_nao_base = list(range(A.shape[1] - m))

            B = A[:, indices_base]
            N = A[:, indices_nao_base]

            cn_f1 = [0.0] * N.shape[1]

            solucao_f1, valor_f1 = simplex_fase2(B, N, b, cn_f1, nomes_base, nomes_nao_base, A)

            if solucao_f1 is None or valor_f1 > 1e-6:
                print("\n Problema inviavel.")
                return

            # Se ainda tiver artificial ativa, não existe solução viável
            for nome in nomes_base:
                if nome.startswith('a') and solucao_f1.get(nome, 0.0) > 1e-6:
                    print(" Problema inviavel: variavel artificial ainda ativa apos Fase 1 forçada.")
                    return

            print("\n Solução viável obtida após Fase 1 forçada. Pronto para Fase 2.")

            cn = np.array([c[i] for i in indices_nao_base])

            simplex_fase2(B, N, b, cn, nomes_base, nomes_nao_base, A)
            return

        # Se base já era válida, inicia Fase 2 direto
        cn = np.array([c[nomes_variaveis.index(nome)] for nome in nomes_nao_base])
        simplex_fase2(B, N, b, cn, nomes_base, nomes_nao_base, A)

if __name__ == "__main__":
    main()

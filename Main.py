#MAIN
from Parser import ler_modelo_matematico_txt
from Simplex import simplex_fase2
import numpy as np

def construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais):
    """
    Constrói as matrizes B e N, além dos vetores cb e cn,
    necessários para executar a Fase 1 do método Simplex.
    
    A base inicial é composta pelas variáveis artificiais.
    """
    B_cols = []
    N_cols = []
    cb = []
    cn = []

    # Monta matriz da base B e vetor cb: colunas correspondentes a nomes_base
    for nome in nomes_base:
        # Encontra índice da variável no A, que pode estar em nomes_base ou nomes_nao_base
        if nome in nomes_base:
            idx = nomes_base.index(nome)
        else:
            idx = nomes_nao_base.index(nome) + len(nomes_base)
        B_cols.append(A[:, idx])
        cb.append(1.0 if nome in artificiais else 0.0)

    # Monta matriz não-base N e vetor cn: colunas correspondentes a nomes_nao_base
    for nome in nomes_nao_base:
        idx = nomes_nao_base.index(nome) + len(nomes_base)
        N_cols.append(A[:, idx])
        cn.append(0.0)

    B = np.column_stack(B_cols)
    N = np.column_stack(N_cols)
    cn = np.array(cn)

    return B, N, b, cn, nomes_base.copy(), nomes_nao_base.copy()


def main():
    dados = ler_modelo_matematico_txt("entrada.txt")
    A = dados["A"]
    b = dados["b"]
    c = dados["c"]
    nomes_base = dados["nomes_base"]
    nomes_nao_base = dados["nomes_nao_base"]
    artificiais = dados["variaveis_artificiais"]
    nomes_variaveis = dados["nomes_variaveis"]

    if artificiais:
        print(">>> FASE 1: resolvendo problema artificial...")

        B, N, tb, cn_f1, base, nao_base = construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais)

        solucao_f1, valor_f1 = simplex_fase2(B, N, tb, cn_f1, base, nao_base)

        if solucao_f1 is None or valor_f1 > 1e-6:
            print("\n Problema inviavel.")
            return

        for nome in base:
            if nome in artificiais and solucao_f1.get(nome, 0.0) > 1e-6:
                print(" Problema inviavel: variavel artificial ainda ativa na base com valor > 0.")
                return

        print("\n Solução viavel encontrada na Fase I.")
        print(">>> FASE 2: otimizando função original...")

        # Monta B e N para fase 2 a partir dos nomes de base e não-base
        indices_base = [nomes_variaveis.index(nome) for nome in base]
        indices_nao_base = [nomes_variaveis.index(nome) for nome in nao_base]

        B = A[:, indices_base]
        N = A[:, indices_nao_base]

        # Custos da função original para as variáveis não básicas na ordem correta
        cn = np.array([c[i] for i in indices_nao_base])

        simplex_fase2(B, N, b, cn, base, nao_base)

    else:
        print(">>> FASE 2 DIRETA")

        m = len(b)

        # Monta base e não-base de acordo com nomes, para garantir ordem correta
        indices_base = [nomes_variaveis.index(nome) for nome in nomes_base]
        indices_nao_base = [nomes_variaveis.index(nome) for nome in nomes_nao_base]

        B = A[:, indices_base]
        N = A[:, indices_nao_base]

        try:
            np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("Base singular. Alternando para Fase 1...")

            artificiais = [f"a{i+1}" for i in range(m)]
            nomes_base = artificiais.copy()
            nomes_nao_base = [f"x{i+1}" for i in range(A.shape[1])]

            A = np.hstack([A, np.identity(m)])
            c = np.concatenate([c, np.ones(m)])

            indices_base = list(range(A.shape[1] - m, A.shape[1]))
            indices_nao_base = list(range(A.shape[1] - m))

            B = A[:, indices_base]
            N = A[:, indices_nao_base]

            cn_f1 = [0.0] * N.shape[1]

            solucao_f1, valor_f1 = simplex_fase2(B, N, b, cn_f1, nomes_base, nomes_nao_base)

            if solucao_f1 is None or valor_f1 > 1e-6:
                print("\n Problema inviavel.")
                return

            for nome in nomes_base:
                if nome.startswith('a') and solucao_f1.get(nome, 0.0) > 1e-6:
                    print(" Problema inviavel: variavel artificial ainda ativa apos Fase 1 forçada.")
                    return

            print("\n Solução viável obtida após Fase 1 forçada. Pronto para Fase 2.")

            todas = nomes_base + nomes_nao_base
            # Aqui nomes_nao_base não mudou e c já está atualizado para artificiais

            cn = np.array([c[i] for i in indices_nao_base])

            simplex_fase2(B, N, b, cn, nomes_base, nomes_nao_base)
            return

        cn = np.array([c[nomes_variaveis.index(nome)] for nome in nomes_nao_base])
        simplex_fase2(B, N, b, cn, nomes_base, nomes_nao_base)

if __name__ == "__main__":
    main()

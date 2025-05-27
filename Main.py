# main.py
from Parser import ler_modelo_matematico_txt
from Simplex import simplex_fase2
import numpy as np

def construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais):
    B = []
    N = []
    cb = []
    cn = []

    for i, nome in enumerate(nomes_base):
        B.append(A[:, i])
        cb.append(1.0 if nome in artificiais else 0.0)

    for i, nome in enumerate(nomes_nao_base):
        col_idx = len(nomes_base) + i
        N.append(A[:, col_idx])
        cn.append(0.0)

    B = np.column_stack(B)
    N = np.column_stack(N)

    return B, N, b, np.array(cn), nomes_base.copy(), nomes_nao_base.copy()

def substituir_artificiais_da_base(A, b, nomes_base, nomes_nao_base, artificiais):
    m = len(b)
    nomes_todas = nomes_base + nomes_nao_base

    # Remove colunas artificiais de A e c
    indices_artificiais = [i for i, nome in enumerate(nomes_todas) if nome in artificiais]
    A_sem_artificiais = np.delete(A, indices_artificiais, axis=1)

    # Remove artificiais dos nomes
    nomes_base_sem_art = [n for n in nomes_base if n not in artificiais]
    nomes_nao_base_sem_art = [n for n in nomes_nao_base if n not in artificiais]
    nomes_restantes = nomes_base_sem_art + nomes_nao_base_sem_art

    # Tenta completar base com variáveis restantes
    B_cols = []
    novos_nomes_base = []
    usados = set()

    for nome in nomes_base_sem_art:
        idx = nomes_restantes.index(nome)
        B_cols.append(A_sem_artificiais[:, idx])
        novos_nomes_base.append(nome)
        usados.add(idx)

    for i, nome in enumerate(nomes_nao_base_sem_art):
        if len(B_cols) >= m:
            break
        idx = len(nomes_base_sem_art) + i
        if idx in usados:
            continue
        col = A_sem_artificiais[:, idx]
        if np.linalg.matrix_rank(B_cols + [col]) == len(B_cols) + 1:
            B_cols.append(col)
            novos_nomes_base.append(nome)
            usados.add(idx)

    if len(B_cols) < m:
        raise ValueError("Não foi possível substituir todas as variáveis artificiais da base por variáveis viáveis.")

    N_cols = []
    novos_nomes_nao_base = []
    for i, nome in enumerate(nomes_restantes):
        if i not in usados:
            N_cols.append(A_sem_artificiais[:, i])
            novos_nomes_nao_base.append(nome)

    B = np.column_stack(B_cols)
    N = np.column_stack(N_cols)

    return B, N, novos_nomes_base, novos_nomes_nao_base

def main():
    dados = ler_modelo_matematico_txt("entrada.txt")
    A = dados["A"]
    b = dados["b"]
    c = dados["c"]
    nomes_base = dados["nomes_base"]
    nomes_nao_base = dados["nomes_nao_base"]
    artificiais = dados["variaveis_artificiais"]

    if artificiais:
        print(">>> FASE 1: resolvendo problema artificial...")
        B, N, tb, cn_f1, base, nao_base = construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais)
        solucao_f1, valor_f1 = simplex_fase2(B, N, tb, cn_f1, base, nao_base)

        if solucao_f1 is None or valor_f1 > 1e-6:
            print("\n Problema inviável.")
            return

        print("\n Solução viável encontrada na Fase I. Iniciando Fase II...")

        try:
            B, N, base, nao_base = substituir_artificiais_da_base(A, b, nomes_base, nomes_nao_base, artificiais)
        except ValueError as e:
            print("Erro ao substituir variáveis artificiais:", e)
            return

        m = len(b)
        cn = c[[i for i, nome in enumerate(base + nao_base) if nome in nao_base]]
        simplex_fase2(B, N, b, cn, base, nao_base)

    else:
        print(">>> FASE 2 DIRETA")
        m = len(b)
        B = A[:, :m]

        try:
            np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("\n Base inicial é singular. Ativando Fase I forçadamente...")

            artificiais = [f"a{i+1}" for i in range(m)]
            nomes_base = artificiais.copy()
            nomes_nao_base = [f"x{i+1}" for i in range(A.shape[1] - m)]
            A = np.hstack([A, np.identity(m)])
            c = np.concatenate([c, np.ones(m)])

            B = A[:, -m:]
            N = A[:, :-m]
            cn_f1 = [0.0] * N.shape[1]

            simplex_fase2(B, N, b, cn_f1, nomes_base, nomes_nao_base)
            return

        N = A[:, m:]
        cn = c[m:]
        simplex_fase2(B, N, b, cn, nomes_base, nomes_nao_base)

if __name__ == "__main__":
    main()

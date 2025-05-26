
from Parser import ler_modelo_matematico_txt
from Simplex import simplex_fase2
import numpy as np

def construir_fase1(A, b, nomes_base, nomes_nao_base, artificiais):
    m, n = A.shape
    B = []
    N = []
    cb = []
    cn = []

    base_indices = []
    nao_base_indices = []

    for i, nome in enumerate(nomes_base):
        B.append(A[:, i])
        cb.append(1.0 if nome in artificiais else 0.0)
        base_indices.append(i)

    for i, nome in enumerate(nomes_nao_base):
        col_idx = len(nomes_base) + i
        N.append(A[:, col_idx])
        cn.append(0.0)
        nao_base_indices.append(col_idx)

    B = np.column_stack(B)
    N = np.column_stack(N)

    return B, N, b, np.array(cn), nomes_base.copy(), nomes_nao_base.copy()

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

        indices_validos = [i for i, nome in enumerate(nomes_base) if nome not in artificiais]
        A_sem_artificiais = A[:, :len(c)]
        B = A_sem_artificiais[:, indices_validos]
        N = A_sem_artificiais[:, [i for i in range(A_sem_artificiais.shape[1]) if i not in indices_validos]]
        base = [nomes_base[i] for i in indices_validos]
        nao_base = [v for v in nomes_nao_base if v not in artificiais]
        tb = b
        cn = c[[i for i in range(len(c)) if i not in indices_validos]]

        simplex_fase2(B, N, tb, cn, base, nao_base)
    else:
        print(">>> FASE 2 DIRETA")
        m = len(b)
        B = A[:, :m]
        N = A[:, m:]
        cn = c[m:]
        simplex_fase2(B, N, b, cn, nomes_base, nomes_nao_base)

if __name__ == "__main__":
    main()

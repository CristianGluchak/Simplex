import numpy as np

def simplex_fase2(B, N, tb, cn, nomes_base, nomes_nao_base):
    m = B.shape[0],
    cb = np.zeros(m)
    nomes_todas = nomes_base + nomes_nao_base
    xb = None

    while True:
        B_inv = np.linalg.inv(B)
        xb = B_inv @ tb
        z_c = cb @ B_inv @ N - cn

        print("\nSolucao basica xb:", xb)
        print("Custos reduzidos (z - c):", z_c)

        if np.all(z_c <= 0):
            valor_otimo = cb @ xb
            print("\n Solucao otima encontrada!")
            print("Valor otimo da funcao objetivo:", valor_otimo)

            solucao_completa = {var: 0 for var in nomes_todas}
            for i, var in enumerate(nomes_base):
                solucao_completa[var] = xb[i]

            print("\nSolucao completa:")
            for var in sorted(solucao_completa.keys()):
                print(f"{var} = {solucao_completa[var]}")
            return solucao_completa, valor_otimo

        j_entrada = np.argmax(z_c)
        Aj = N[:, j_entrada]
        d = B_inv @ Aj

        print(f"\nVariavel de entrada: {nomes_nao_base[j_entrada]}")
        print("Direcao d:", d)

        if np.all(d <= 0):
            print(" Problema ilimitado (sem solução finita).")
            return None, None

        razoes = [xb[i] / d[i] if d[i] > 0 else np.inf for i in range(len(d))]
        i_saida = np.argmin(razoes)

        print(f"Variavel de saida: {nomes_base[i_saida]}")

        B[:, i_saida] = Aj
        cb[i_saida] = cn[j_entrada]
        nomes_base[i_saida], nomes_nao_base[j_entrada] = nomes_nao_base[j_entrada], nomes_base[i_saida]

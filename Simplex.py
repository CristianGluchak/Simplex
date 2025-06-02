import numpy as np

def simplex_fase2(B, N, tb, cn, nomes_base, nomes_nao_base):
    m = B.shape[0]  
    nomes_todas = nomes_base + nomes_nao_base

    # Mapeia os custos das variáveis (0 para base, cn para não base)
    c_dict = {nome: 0.0 for nome in nomes_base}
    for nome, custo in zip(nomes_nao_base, cn):
        c_dict[nome] = custo

    cb = np.array([c_dict[nome] for nome in nomes_base])

    while True:
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print(" Base singular durante Fase 2. Problema mal condicionado.")
            return None, None

        xb = B_inv @ tb  # solução básica atual

        # Custos reduzidos z - c para variáveis não básicas
        z_c = cb @ B_inv @ N - cn

        print("\n Solucao basica xb:", xb)
        print("Base atual:", nomes_base)
        print("Custos reduzidos (z - c):", z_c)

        if np.all(z_c <= 1e-8):  # Critério de otimalidade para max
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
            print("Problema ilimitado (sem solução finita).")
            return None, None

        razoes = [xb[i] / d[i] if d[i] > 0 else np.inf for i in range(len(d))]
        i_saida = np.argmin(razoes)

        print(f"Variavel de saida: {nomes_base[i_saida]}")

        # Atualiza base e custos
        B[:, i_saida] = Aj
        cb[i_saida] = cn[j_entrada]
        nomes_base[i_saida], nomes_nao_base[j_entrada] = nomes_nao_base[j_entrada], nomes_base[i_saida]

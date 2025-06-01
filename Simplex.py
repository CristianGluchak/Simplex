import numpy as np

def simplex_fase2(B, N, tb, cn, nomes_base, nomes_nao_base):
    """
    Executa a Fase 2 do método Simplex para um problema linear na forma padrão.

    Parâmetros:
        B (np.ndarray): matriz das variáveis básicas (colunas da base)
        N (np.ndarray): matriz das variáveis não básicas (colunas fora da base)
        tb (np.ndarray): vetor lado direito (termos independentes)
        cn (np.ndarray): custos associados às variáveis não básicas
        nomes_base (list): nomes das variáveis na base
        nomes_nao_base (list): nomes das variáveis fora da base

    Retorna:
        - dicionário com a solução ótima
        - valor da função objetivo
    """
    m = B.shape[0]  # número de restrições
    nomes_todas = nomes_base + nomes_nao_base

    # Mapeia todos os custos: 0 para variáveis básicas, cn para variáveis não básicas
    c_dict = {nome: val for nome, val in zip(nomes_todas, [0.0]*len(nomes_base) + list(cn))}

    # Vetor de custos da base (cb), extraído do dicionário c_dict
    cb = np.array([c_dict.get(nome, 0.0) for nome in nomes_base])

    xb = None  # Solução básica inicial

    while True:
        # Inversão da base para cálculo da solução atual
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print(" Base singular durante Fase 2. Problema mal condicionado.")
            return None, None

        xb = B_inv @ tb  # solução básica atual

        # Custos reduzidos: z - c
        z_c = cb @ B_inv @ N - cn

        print("\n Solucao basica xb:", xb)
        print("Base atual:", nomes_base)
        print("Custos reduzidos (z - c):", z_c)

        # Critério de otimalidade: se todos os custos reduzidos são ≤ 0 (para min), parar
        if np.all(z_c <= 1e-8):  # tolerância numérica
            valor_otimo = cb @ xb  # valor da função objetivo
            print("\n Solucao otima encontrada!")
            print("Valor otimo da funcao objetivo:", valor_otimo)

            # Reconstrói a solução completa com todas as variáveis (base + fora da base)
            solucao_completa = {var: 0 for var in nomes_todas}
            for i, var in enumerate(nomes_base):
                solucao_completa[var] = xb[i]

            print("\nSolucao completa:")
            for var in sorted(solucao_completa.keys()):
                print(f"{var} = {solucao_completa[var]}")
            return solucao_completa, valor_otimo

        # Seleciona a variável de entrada (com maior custo reduzido)
        j_entrada = np.argmax(z_c)
        Aj = N[:, j_entrada]        # Coluna associada à variável que entra
        d = B_inv @ Aj              # Direção de movimento no espaço das soluções

        print(f"\nVariavel de entrada: {nomes_nao_base[j_entrada]}")
        print("Direcao d:", d)

        # Se direção ≤ 0, então problema é ilimitado (não existe pivô válido)
        if np.all(d <= 0):
            print("Problema ilimitado (sem solução finita).")
            return None, None

        # Regra da razão mínima (Bland): determina a variável que sairá da base
        razoes = [xb[i] / d[i] if d[i] > 0 else np.inf for i in range(len(d))]
        i_saida = np.argmin(razoes)

        print(f"Variavel de saida: {nomes_base[i_saida]}")

        # Atualização da base: substitui coluna e nome
        B[:, i_saida] = Aj
        cb[i_saida] = cn[j_entrada]
        nomes_base[i_saida], nomes_nao_base[j_entrada] = nomes_nao_base[j_entrada], nomes_base[i_saida]

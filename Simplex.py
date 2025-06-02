import numpy as np

def simplex_fase2(B, N, tb, cn, nomes_base, nomes_nao_base):
    """
    Executa a Fase 2 do método Simplex para um problema de Programação Linear já viável.
    
    Parâmetros:
        B (np.ndarray): matriz das variáveis básicas (colunas da base)
        N (np.ndarray): matriz das variáveis não básicas
        tb (np.ndarray): vetor de termos independentes (lado direito)
        cn (np.ndarray): vetor de custos das variáveis não básicas
        nomes_base (list): nomes das variáveis na base atual
        nomes_nao_base (list): nomes das variáveis fora da base

    Retorno:
        - Dicionário com a solução ótima (valores de todas as variáveis)
        - Valor da função objetivo
    """

    m = B.shape[0]  # número de restrições
    nomes_todas = nomes_base + nomes_nao_base

    # Cria dicionário de custos: 0 para variáveis básicas, cn para as não básicas
    c_dict = {nome: 0.0 for nome in nomes_base}
    for nome, custo in zip(nomes_nao_base, cn):
        c_dict[nome] = custo

    # Constrói vetor de custos cb com base na ordem das variáveis na base
    cb = np.array([c_dict[nome] for nome in nomes_base])

    while True:
        # Calcula inversa da base para encontrar solução atual
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print(" Base singular durante Fase 2. Problema mal condicionado.")
            return None, None

        xb = B_inv @ tb  # solução básica atual (valores das variáveis básicas)

        # Calcula os custos reduzidos z_j - c_j para variáveis não básicas
        z_c = cb @ B_inv @ N - cn

        print("\n Solucao basica xb:", xb)
        print("Base atual:", nomes_base)
        print("Custos reduzidos (z - c):", z_c)

        # Critério de otimalidade: se todos z - c <= 0, a solução é ótima
        if np.all(z_c <= 1e-8):  # tolerância para zero numérico
            valor_otimo = cb @ xb
            print("\n Solucao otima encontrada!")
            print("Valor otimo da funcao objetivo:", valor_otimo)

            # Reconstrói vetor completo da solução (base + não-base)
            solucao_completa = {var: 0 for var in nomes_todas}
            for i, var in enumerate(nomes_base):
                solucao_completa[var] = xb[i]

            print("\nSolucao completa:")
            for var in sorted(solucao_completa.keys()):
                print(f"{var} = {solucao_completa[var]}")
            return solucao_completa, valor_otimo

        # Escolhe a variável de entrada (com maior z - c)
        j_entrada = np.argmax(z_c)
        Aj = N[:, j_entrada]        # Coluna da variável que vai entrar
        d = B_inv @ Aj              # Direção de movimento no espaço de soluções

        print(f"\nVariavel de entrada: {nomes_nao_base[j_entrada]}")
        print("Direcao d:", d)

        # Se todos os elementos de d ≤ 0, o problema é ilimitado
        if np.all(d <= 0):
            print("Problema ilimitado (sem solução finita).")
            return None, None

        # Regra da razão mínima para encontrar a variável de saída
        razoes = [xb[i] / d[i] if d[i] > 0 else np.inf for i in range(len(d))]
        i_saida = np.argmin(razoes)

        print(f"Variavel de saida: {nomes_base[i_saida]}")

        # Atualiza a base: substitui variável de saída pela de entrada
        B[:, i_saida] = Aj
        cb[i_saida] = cn[j_entrada]

        # Troca os nomes das variáveis
        nomes_base[i_saida], nomes_nao_base[j_entrada] = nomes_nao_base[j_entrada], nomes_base[i_saida]

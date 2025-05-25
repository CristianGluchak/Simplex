import numpy as np

def ler_entrada_com_nomes(caminho):
    with open(caminho, 'r') as f:
        linhas = f.readlines()
 
    B, N, tb, cn = [], [], [], []
    nomes_base, nomes_nao_base = [], []
    secao = None

    for linha in linhas:
        linha = linha.strip()
        if not linha or linha.startswith('#'):
            continue

        if linha.endswith(':'):
            secao = linha[:-1]
            continue

        if secao == 'B':
            B.append(list(map(float, linha.split())))
        elif secao == 'N':
            N.append(list(map(float, linha.split())))
        elif secao == 'tb':
            tb = list(map(float, linha.split()))
        elif secao == 'cn':
            cn = list(map(float, linha.split()))
        elif secao == 'nomes_base':
            nomes_base = linha.split()
        elif secao == 'nomes_nao_base':
            nomes_nao_base = linha.split()

    return (np.array(B), np.array(N), 
            np.array(tb), np.array(cn), 
            nomes_base, nomes_nao_base)


def simplex_fase2(B, N, tb, cn, nomes_base, nomes_nao_base):
    m, n = B.shape[0], N.shape[1]
    
    cb = np.zeros(m)
    
    # Mantém a ordem das variáveis (base + não-base)
    nomes_todas = nomes_base + nomes_nao_base
    variaveis = nomes_base.copy()
    xb = None

    while True:
        B_inv = np.linalg.inv(B)
        xb = B_inv @ tb
        z_c = cb @ B_inv @ N - cn

        print("\nSolucao basica xb:", xb)
        print("Custos reduzidos (z - c):", z_c)

        # Verifica otimalidade
        if np.all(z_c <= 0):
            valor_otimo = cb @ xb
            print("\n Solucao otima encontrada!")
            print("Valor otimo da funcao objetivo:", valor_otimo)

            # Reconstrói vetor de solução completa
            solucao_completa = {var: 0 for var in nomes_todas}
            for i, var in enumerate(nomes_base):
                solucao_completa[var] = xb[i]

            print("\nSolucao completa:")
            for var in sorted(solucao_completa.keys()):
                print(f"{var} = {solucao_completa[var]}")
            return solucao_completa, valor_otimo

        # Seleciona variável de entrada (maior custo reduzido)
        j_entrada = np.argmax(z_c)
        Aj = N[:, j_entrada]
        d = B_inv @ Aj

        print(f"\nVariavel de entrada: {nomes_nao_base[j_entrada]}")
        print("Direcao d:", d)

        if np.all(d <= 0):
            print(" Problema ilimitado (sem solução finita).")
            return None, None

        # Razão mínima
        razoes = [xb[i] / d[i] if d[i] > 0 else np.inf for i in range(len(d))]
        i_saida = np.argmin(razoes)

        print(f"Variavel de saida: {nomes_base[i_saida]}")

        # Pivot: atualizar base
        B[:, i_saida] = Aj
        cb[i_saida] = cn[j_entrada]

        # Trocar nomes
        nomes_base[i_saida], nomes_nao_base[j_entrada] = nomes_nao_base[j_entrada], nomes_base[i_saida]

# Leitura do arquivo
arquivo = 'C:\Simplex\initialDatas.txt'
B, N, tb, cn, nomes_base, nomes_nao_base = ler_entrada_com_nomes(arquivo)

# Executa o simplex fase 2
simplex_fase2(B, N, tb, cn, nomes_base, nomes_nao_base)

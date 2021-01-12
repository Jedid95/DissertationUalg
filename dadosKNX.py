#Script para trabalhar com os dados .csv do software ETS5 do KNX
import pandas

file_name = 'enderecos.csv'

def lerFicheiro(opcao):

    if opcao == 1:
        df = pandas.read_csv(file_name,encoding = "ISO-8859-1", sep=';')
        print(df)
    elif opcao == 2:
        df = pandas.read_csv(file_name,encoding = "ISO-8859-1", sep=';', usecols=['Sub'])
        print(df)
    elif opcao == 3:
        df = pandas.read_csv(file_name,encoding = "ISO-8859-1", sep=';', usecols=['Address'])
        print(df)
    

def menu():
    opcao=int(input('''
                        Escolha uma opção:
                        1 - Mostrar todo ficheiro
                        2 - Mostrar nomes dos dispositivos
                        3 - Mostrar endereços dos dispositivos
                        4 - Fechar Menu
                        Escolha:  '''))
    if opcao == 1:
        print("Apresentar todo ficheiro")
        lerFicheiro(1)
    elif opcao == 2:
        print("Apresentar nomes dos dispositivos")
        lerFicheiro(2)
    elif opcao == 3:
        print("Apresentar endereços dos dispositivos")
        lerFicheiro(3)
    elif opcao == 4:
        exit()
    else:
        print("Este número não está nas alternativas, tente novamente")
        menu()

while True:
    menu()
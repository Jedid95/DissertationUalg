#Import for voice speak
import pyttsx3

#Init voice speak
voice = pyttsx3.init()

'''Movements
0 - Active/Disable | 1 - Next | 2 - Previous | 3 - Increase | 4 - Decrease | 5 - Favorite'''

'''Type Devices
0 - Iluminacao | 1 - Estores | 2 - Climatizacao | 3 - Tomadas | 4 - Termostatos | 5 - Piso Radiante '''


family = ['Iluminacao','Estores', 'Climatizacao','Tomadas','Termostatos','Piso Radiante']
devices_ilum = ['LAMP1','LAMP2','LAMP3']
devices_esto =['EST1','EST2']
devices_clim =['C1','C2','C3']
devices_toma =['TOMA1','TOMA2','TOMA3','TOMA4']
devices_termo =['TERM1']
devices_piso =['PISO1']

operations = ['On/Off','Incrementar','decrementar']

#Variables menu_one
global posicao_one, tipo_one, tamanho_family 
posicao_one = 0
tipo_one = 0
tamanho_family = len(family)-1

#Variables menu_two
global tam_ilum, tam_esto, tam_clim, tam_toma, tam_termo, tam_piso
tam_ilum = len(devices_ilum)-1
tam_esto = len(devices_esto)-1
tam_clim = len(devices_clim)-1
tam_toma = len(devices_toma)-1
tam_termo = len(devices_termo)-1
tam_piso = len(devices_piso)-1
global posicao_two
posicao_two = 0


def menu_one():
    global posicao_one, tipo_one, tamanho_family
    voice.say(family[posicao_one] + " - Movement: ")
    voice.runAndWait()
    opcao=int(input(family[posicao_one] + ' - Movement: ' ))


    if opcao == 5:
        print('Favorite Scenary')
        voice.say("Favorite Scenary")
        voice.runAndWait()

    elif opcao == 0:
        print(family[posicao_one] + ' Select')
        menu_two(tipo_one,posicao_one)

    elif opcao == 1:
        if posicao_one == tamanho_family:
            posicao_one = 0
            tipo_one = 0
            menu_one()
        else:
            posicao_one +=1
            tipo_one += 1
            menu_one()
    else:
        print("Este número não está nas alternativas, tente novamente")
        posicao_one = 0
        tipo_one = 0
        voice.say("Este número não está nas alternativas, tente novamente")
        voice.runAndWait()
        menu_one()


def menu_two(t,p):
    global posicao_two
    if p ==0: #Iluminação
        voice.say(devices_ilum[posicao_two] + " - Movement: ")
        voice.runAndWait()
        opcao=int(input('Movement: ' ))
    elif p == 1: #Estores
        voice.say(devices_esto[posicao_two] + " - Movement: ")
        voice.runAndWait()
        opcao=int(input('Movement: ' ))
    elif p == 2: #Climatizacao
        voice.say(devices_clim[posicao_two] + " - Movement: ")
        voice.runAndWait()
        opcao=int(input('Movement: ' ))
    elif p == 3: #Tomadas
        voice.say(devices_toma[posicao_two] + " - Movement: ")
        voice.runAndWait()
        opcao=int(input('Movement: ' ))
    elif p == 4: #Termostatos
        voice.say(devices_termo[posicao_two] + " - Movement: ")
        voice.runAndWait()
        opcao=int(input('Movement: ' ))
    elif p == 2: #PisoRadiante
        voice.say(devices_piso[posicao_two] + " - Movement: ")
        voice.runAndWait()
        opcao=int(input('Movement: ' ))

    if opcao == 1:
        if p==0: #Iluminação
            if posicao_two == tam_ilum:
                posicao_two = 0
            else:
                posicao_two += 1
            menu_two(t,p)
        if p==1: #Estores
            if posicao_two == tam_esto:
                posicao_two = 0
            else:
                posicao_two += 1
            menu_two(t,p)
        if p==2: #Climatizacao
            if posicao_two == tam_clim:
                posicao_two = 0
            else:
                posicao_two += 1
            menu_two(t,p)
        if p==3: #Tomadas
            if posicao_two == tam_toma:
                posicao_two = 0
            else:
                posicao_two += 1
            menu_two(t,p)
        if p==4: #Termostatos
            if posicao_two == tam_termo:
                posicao_two = 0
            else:
                posicao_two += 1
            menu_two(t,p)
        if p==5: #PisoRadiante
            if posicao_two == tam_piso:
                posicao_two = 0
            else:
                posicao_two += 1
            menu_two(t,p)

file_name = "ETS_enderecos.xls"
import pandas as pd
df = pd.read_excel(io=file_name)
print(df)
menu_one()
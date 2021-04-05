import PySimpleGUI as sg
import pandas as pd

file_name = "ETS_enderecos.xls"
df = pd.read_excel(io=file_name)

types_devices = []

sg.theme('Reddit')
#theme_name_list = sg.theme_list()
#print(theme_name_list)

# All the stuff inside your window.
layout = [  [sg.Text('Add device types equally to what is in the ETS file')],
            [sg.Text('Enter with Type Device'), sg.InputText()],
            [sg.Button('Add'), sg.Button('Finish'), sg.Button('Cancel')] ]

# Create the Window
window = sg.Window('Menu Configuration', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Finish' or event == 'Cancel': # if user closes window or clicks cancel
        break

    if event == 'Add':
        confirm = sg.PopupOKCancel('Add this type？', title='Confirm')
        if confirm == 'OK':
            types_devices.append(values[0])

    #print('You entered ', values[0])
    #print(types_devices)

window.close()

print(types_devices)
#teste = df.query('Type == "Atuadores"').head()
#print(teste['Type'])
# PYTHON 3
# START OF PROGRAM INFORMATION
#-------------------------------------------
# analyze_stock.py
    # Author: Sharv Save
    # Purpose:
        # main program
    # High level description:
        # main program
# END OF PROGRAM INFORMATION
#-------------------------------------------

import PySimpleGUI as sg
import RNN

# define function get_user_input() - user input
def get_user_input():
    print("\ninside get_user_input")

    # create a pysimplegui interface to gather input filenames
    sg.theme('Topanga')
    layout = [
        [sg.Text(' ')],
        [sg.Text('Please enter the ticker symbol of the stock to be analyzed!')],
        [sg.Text('TICKER'), sg.InputText(size=(60, 60), key='TICKER')],
        [sg.Text(' ')],
        [sg.Submit(), sg.Cancel()]
    ]

    window = sg.Window('ANALYZE STOCK', layout)
    event, values = window.read()
    window.close()

    if event == "Cancel":
        exit("user cancelled the run!")

    print("\tUser selected TICKER:", values['TICKER'])
    ticker = str(values['TICKER']).strip().upper()
    print("\tUser selected TICKER:", ticker)

    print("\tget_user_input complete")
    return ticker
# end of function

# start of the main body of the program #
def my_main():
    print("\n++-->> start of ANALYZE STOCK program")
    
    # step 1 - get ticker symbol from the user
    ticker = get_user_input()
    # step 4 - conduct analysis 3
    analysis_THREE = RNN.Analysis3(ticker)
    
    if not analysis_THREE.perform_analysis3():
        exit('there was a problem with Analysis3.perform_analysis3')
    # end if

    print("++-->> successful end of ANALYZE STOCK program\n")
# end of function

if __name__ == "__main__":
    my_main()
# end of if else

# END OF FILE
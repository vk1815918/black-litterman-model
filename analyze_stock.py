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
import Analysis1v3
import Analysis2v3
import Analysis3v3

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
    
    # step 2 - conduct analysis 1
    analysis_ONE = Analysis1v3.Analysis1(ticker)
    
    if not analysis_ONE.perform_analysis1():
        exit('there was a problem with Analysis1.perform_analysis1')
    else:
        fscore_analysis, fscore = analysis_ONE.getFeedback()
        print('\tfscore_analysis::\n\t', fscore_analysis)
        print('\tfscore::\n\t', fscore)
    # end if
    
    # step 3 - conduct analysis 2
    analysis_TWO = Analysis2v3.Analysis2(ticker)

    if not analysis_TWO.perform_analysis2():
        exit('there was a problem with Analysis2.perform_analysis2')
    # end if

    # step 4 - conduct analysis 3
    analysis_THREE = Analysis3v3.Analysis3(ticker)
    
    if not analysis_THREE.perform_analysis3():
        exit('there was a problem with Analysis3.perform_analysis3')
    # end if

    print("++-->> successful end of ANALYZE STOCK program\n")
# end of function

if __name__ == "__main__":
    my_main()
# end of if else

# END OF FILE
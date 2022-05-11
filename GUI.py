'''
This code has been written by Oz Kedem in 2022 for a computer science finals project
this file specifically is about the GUI.
'''

#importing the GUI Library
from tkinter import *
import NeuralNetDiabetes_OzKedem
global inputArray


def GUI():

    mainWindows = Tk()
    mainWindows.title('NeuralNetDiabetes_OzKedem')

    #Creating the upper-centered text:
    topLabel = Label(mainWindows, text="Please Enter The Inputs As Requested below", justify='center')
    topLabel.pack(padx=10)

    #Creating The TextBoxes&Labels:
    lblArray = Label(mainWindows, text="Array Here:")
    lblArray.pack()
    lblOrder = Label(mainWindows, text=
    "In Such Order: [Pregnancies, Glucose, Blood Pressure, Skin thickness,Insulin, BMI, Diabetes Function, Age]")
    lblOrder.pack()

    txtArray = Text(mainWindows, width=50 , height =1)
    txtArray.pack()



    #Creating The General Input Button:
    btnInput = Button(mainWindows, text="Run The Testing", command=lambda: getInput())
    btnInput.pack(pady=10, side=BOTTOM)

    def getInput():
        inputValue = txtArray.get("1.0", "end-1c")
        print(inputValue)

        #SoftCoding
        inputArray = list(map(float, inputValue.split()))

        #HardCoding - DO NOT USE ALL THE TIME
        #inputArray = [5,166,72,19,175,25.8,0.587,51]

        #If - Input Is Legal, Proceed
        #If not - Show an Error Message
        NeuralNetDiabetes_OzKedem.retPrediction(inputArray) #Assuming the input is right

        Answer = NeuralNetDiabetes_OzKedem.retPrediction(inputArray)

        #Printing Wether he is Diabetic Or Not
        if (Answer):
            Prediction = Label(mainWindows, text="Prediction - Is Probably Diabetic")
            Prediction.pack()
            return
        else:
            Prediction = Label(mainWindows, text="Prediction - Isn't Probably Diabetic")
            Prediction.pack()
            return

'''
Examples:
5 166 72 19 175	25.8 0.587 51 ~ A Not So Healthy Women OverAll (1)
0 89 66 23 94 28.1 0.167 15 ~ A Healthy Man OverAll (0)

'''
GUI()
mainloop()

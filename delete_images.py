ok = 0
while ok == 0:
    response = input("Delete taken images? Press 1-delete, 0-save \n")
    if response == '1':
        print(" > Images deleted")
        ok = 1
    elif response == '0':
        print(" > Images saved")
        ok = 1
    else:
        print("Error, please try again")
        print("-----------------------")

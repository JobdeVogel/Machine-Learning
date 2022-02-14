""" THIS FILE CONTAINS GENERAL FUNCTIONS """

def loading(file, amount_of_files):
    percentage = round((file / amount_of_files) * 100, 2)
    print('Loading ' + str(percentage) + '% completed', end="\r")
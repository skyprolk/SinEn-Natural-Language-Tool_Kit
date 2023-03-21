# function to show menu options
def option_menu(options : dict, message = "Please choose an option: "):
    # prints the message
    print(message)
    # loop through options and prints the key and value
    for key, value in options.items():
        print(f"{key}. {value}")
    # user input
    choice = input("\nEnter your choice: ")
    try:
        # convert choice to integer
        choice = int(choice)
        # check if choice is in options
        if choice in options:
            return choice
        else:
            # if choice not in options raise error
            raise ValueError
    except ValueError:
        # if error print invalid choice and call function again
        print("Invalid choice. Please try again.\n")
        return option_menu(options)
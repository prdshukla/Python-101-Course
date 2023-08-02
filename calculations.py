def order_numbers(number1, number2):
    if (number1 > number2):
        bigger = number1
        smaller = number2
    else:
        bigger = number2
        smaller = number1
    return smaller, bigger


print (order_numbers(4,2))







def to_celcius(temp):
    return (temp-32)*5/9



print (to_celcius(453))


for left in range(9):
    for right in range(left,9):
        print("["+str(right) + "|" +str(left) + "]", end=" ")
    print()